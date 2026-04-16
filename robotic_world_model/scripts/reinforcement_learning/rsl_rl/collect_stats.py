"""Collect observation/state and action normalization statistics from a trained policy."""

import argparse
import json
import os
import pprint
import sys
from dataclasses import asdict, dataclass

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Collect observation and action statistics from an RSL-RL policy.")
parser.add_argument("--num_steps", type=int, default=5000, help="Number of policy steps to collect.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of real environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument(
    "--stats_output",
    type=str,
    default=None,
    help="Optional JSON output path. Defaults to <log_dir>/stats/<task>_stats.json.",
)
parser.add_argument(
    "--system_dynamics_load_path",
    type=str,
    default=None,
    help="Optional dynamics model load path when using MBPO runners.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, MBPOOnPolicyRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import mbrl.tasks  # noqa: F401


class StreamingMoments:
    """Track per-dimension sample mean/std without storing all samples."""

    def __init__(self, dim: int, device: torch.device):
        self.count = 0
        self.mean = torch.zeros(dim, dtype=torch.float64, device=device)
        self.m2 = torch.zeros(dim, dtype=torch.float64, device=device)

    def update(self, batch: torch.Tensor) -> None:
        if batch.ndim != 2:
            raise ValueError(f"Expected a 2D tensor, got shape {tuple(batch.shape)}.")
        if batch.shape[0] == 0:
            return

        batch = batch.to(dtype=torch.float64)
        batch_count = batch.shape[0]
        batch_mean = batch.mean(dim=0)
        batch_m2 = torch.sum((batch - batch_mean).pow(2), dim=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        self.m2 = self.m2 + batch_m2 + delta.pow(2) * self.count * batch_count / total_count
        self.count = total_count

    def mean_list(self) -> list[float]:
        return self.mean.detach().cpu().tolist()

    def std_list(self, min_std: float = 1.0e-6) -> list[float]:
        if self.count <= 1:
            std = torch.full_like(self.mean, min_std)
        else:
            variance = self.m2 / (self.count - 1)
            std = torch.sqrt(torch.clamp(variance, min=min_std**2))
        return std.detach().cpu().tolist()


@dataclass
class StatsResult:
    task: str
    checkpoint: str
    num_steps: int
    num_samples: int
    state_dim: int
    action_dim: int
    state_mean: list[float]
    state_std: list[float]
    action_mean: list[float]
    action_std: list[float]


def _get_resume_path(task_name: str, agent_cfg: RslRlBaseRunnerCfg) -> str:
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _resolve_real_env_ids(base_env) -> torch.Tensor:
    env_ids_real = getattr(base_env, "env_ids_real", None)
    if env_ids_real is not None:
        return env_ids_real
    return torch.arange(base_env.num_envs, device=base_env.device)


def _get_obs_group(base_env, group_name: str) -> torch.Tensor:
    obs_groups = getattr(base_env, "obs_buf", None)
    if obs_groups is not None and group_name in obs_groups:
        return obs_groups[group_name]

    obs_groups = base_env.observation_manager.compute()
    if group_name not in obs_groups:
        available_groups = ", ".join(obs_groups.keys())
        raise KeyError(f"Observation group '{group_name}' is unavailable. Found groups: {available_groups}")
    return obs_groups[group_name]


def _build_runner(env, agent_cfg: RslRlBaseRunnerCfg, log_dir: str):
    if agent_cfg.class_name == "OnPolicyRunner":
        return OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if agent_cfg.class_name == "DistillationRunner":
        return DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if agent_cfg.class_name == "MBPOOnPolicyRunner":
        agent_cfg.system_dynamics_load_path = (
            args_cli.system_dynamics_load_path
            if args_cli.system_dynamics_load_path is not None
            else agent_cfg.system_dynamics_load_path
        )
        return MBPOOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")


def _load_runner_checkpoint(runner, resume_path: str) -> None:
    try:
        runner.load(resume_path, load_optimizer=False)
    except TypeError:
        runner.load(resume_path)


def _default_stats_output(log_dir: str, task_name: str) -> str:
    safe_task_name = task_name.replace(":", "_")
    return os.path.join(log_dir, "stats", f"{safe_task_name}_stats.json")


def _save_stats(result: StatsResult, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)
    print(f"[INFO] Saved statistics to: {output_path}")


def _print_summary(result: StatsResult) -> None:
    formatter = pprint.PrettyPrinter(width=120, compact=False, sort_dicts=False)
    print("")
    print("[INFO] Statistics collection complete.")
    print(f"[INFO] Samples: {result.num_samples} | State dim: {result.state_dim} | Action dim: {result.action_dim}")
    print("")
    print("# Paste into state normalizer config")
    print("state_data_mean =")
    print(formatter.pformat(result.state_mean))
    print("state_data_std =")
    print(formatter.pformat(result.state_std))
    print("")
    print("# Paste into action normalizer config")
    print("action_data_mean =")
    print(formatter.pformat(result.action_mean))
    print("action_data_std =")
    print(formatter.pformat(result.action_std))


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run a trained policy and collect state/action statistics."""
    task_name = args_cli.task.split(":")[-1]
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)

    requested_real_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if "Visualize" in task_name:
        env_cfg.scene.num_envs = requested_real_envs * 2
    else:
        env_cfg.scene.num_envs = requested_real_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    resume_path = _get_resume_path(task_name, agent_cfg)
    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = _build_runner(env, agent_cfg, log_dir)
    _load_runner_checkpoint(runner, resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    if hasattr(env.unwrapped, "init_imagination_history") and hasattr(agent_cfg, "system_dynamics"):
        env.unwrapped.init_imagination_history(agent_cfg.system_dynamics.history_horizon)

    obs = env.get_observations()
    real_env_ids = _resolve_real_env_ids(env.unwrapped)

    state_example = _get_obs_group(env.unwrapped, "system_state")[real_env_ids]
    action_dim = env.unwrapped.action_manager.total_action_dim
    state_stats = StreamingMoments(dim=state_example.shape[-1], device=env.unwrapped.device)
    action_stats = StreamingMoments(dim=action_dim, device=env.unwrapped.device)

    print(
        f"[INFO] Collecting stats for {args_cli.num_steps} steps "
        f"across {len(real_env_ids)} real environments on {env.unwrapped.device}."
    )

    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            current_state = _get_obs_group(env.unwrapped, "system_state")[real_env_ids]
            current_actions = actions[real_env_ids]
            state_stats.update(current_state)
            action_stats.update(current_actions)
            obs, _, _, _ = env.step(actions)

        if (step + 1) % 100 == 0 or step == 0 or (step + 1) == args_cli.num_steps:
            print(f"[INFO] Collected step {step + 1} / {args_cli.num_steps}")

    result = StatsResult(
        task=task_name,
        checkpoint=resume_path,
        num_steps=args_cli.num_steps,
        num_samples=state_stats.count,
        state_dim=len(state_stats.mean_list()),
        action_dim=len(action_stats.mean_list()),
        state_mean=state_stats.mean_list(),
        state_std=state_stats.std_list(),
        action_mean=action_stats.mean_list(),
        action_std=action_stats.std_list(),
    )

    output_path = args_cli.stats_output or _default_stats_output(log_dir, task_name)
    _save_stats(result, output_path)
    _print_summary(result)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
