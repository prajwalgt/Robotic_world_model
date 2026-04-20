"""Collect state and action normalization statistics from a trained G1 INIT policy."""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Collect state/action normalization stats from a trained policy.")
parser.add_argument("--num_steps", type=int, default=5000, help="Number of policy steps to collect per env.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--output", type=str, default=None, help="JSON output path. Defaults to <log_dir>/normalizer_stats.json.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import mbrl.tasks  # noqa: F401


class StreamingMoments:
    """Welford online algorithm for per-dimension mean and std."""

    def __init__(self, dim: int, device: torch.device):
        self.count = 0
        self.mean = torch.zeros(dim, dtype=torch.float64, device=device)
        self.m2 = torch.zeros(dim, dtype=torch.float64, device=device)

    def update(self, batch: torch.Tensor) -> None:
        batch = batch.to(dtype=torch.float64)
        n = batch.shape[0]
        if n == 0:
            return
        batch_mean = batch.mean(dim=0)
        batch_m2 = torch.sum((batch - batch_mean).pow(2), dim=0)
        if self.count == 0:
            self.count, self.mean, self.m2 = n, batch_mean, batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean += delta * n / total
        self.m2 += batch_m2 + delta.pow(2) * self.count * n / total
        self.count = total

    def mean_list(self) -> list[float]:
        return self.mean.cpu().tolist()

    def std_list(self, min_std: float = 1e-6) -> list[float]:
        if self.count <= 1:
            return [min_std] * self.mean.shape[0]
        variance = self.m2 / (self.count - 1)
        return torch.sqrt(torch.clamp(variance, min=min_std ** 2)).cpu().tolist()


def _get_state(env) -> torch.Tensor:
    """Read 120-dim G1 state directly from robot asset data."""
    robot = env.scene["robot"]
    base_lin_vel = robot.data.root_lin_vel_b                                    # (N, 3)
    base_ang_vel = robot.data.root_ang_vel_b                                    # (N, 3)
    projected_gravity = robot.data.projected_gravity_b                          # (N, 3)
    joint_pos = robot.data.joint_pos - robot.data.default_joint_pos             # (N, 37)
    joint_vel = robot.data.joint_vel                                            # (N, 37)
    joint_torque = robot.data.applied_torque                                    # (N, 37)
    return torch.cat([base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel, joint_torque], dim=-1)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    resume_path = (
        retrieve_file_path(args_cli.checkpoint)
        if args_cli.checkpoint
        else get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    )
    log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    device = env.unwrapped.device
    state_stats = StreamingMoments(dim=120, device=device)
    action_stats = StreamingMoments(dim=37, device=device)

    obs = env.get_observations()
    print(f"[INFO] Collecting stats over {args_cli.num_steps} steps x {env_cfg.scene.num_envs} envs ...")

    for step in range(args_cli.num_steps):
        with torch.inference_mode():
            actions = policy(obs)
            state_stats.update(_get_state(env.unwrapped))
            action_stats.update(actions)
            obs, _, _, _ = env.step(actions)

        if (step + 1) % 500 == 0:
            print(f"[INFO] Step {step + 1} / {args_cli.num_steps}  |  samples so far: {state_stats.count}")

    result = {
        "task": args_cli.task,
        "checkpoint": resume_path,
        "num_samples": state_stats.count,
        "state_mean": state_stats.mean_list(),
        "state_std": state_stats.std_list(),
        "action_mean": action_stats.mean_list(),
        "action_std": action_stats.std_list(),
    }

    output_path = args_cli.output or os.path.join(log_dir, "normalizer_stats.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[INFO] Saved stats to: {output_path}")
    print(f"[INFO] Samples: {state_stats.count} | State dim: 120 | Action dim: 37")
    print("\n# state_mean =")
    print(result["state_mean"])
    print("\n# state_std =")
    print(result["state_std"])
    print("\n# action_mean =")
    print(result["action_mean"])
    print("\n# action_std =")
    print(result["action_std"])

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
