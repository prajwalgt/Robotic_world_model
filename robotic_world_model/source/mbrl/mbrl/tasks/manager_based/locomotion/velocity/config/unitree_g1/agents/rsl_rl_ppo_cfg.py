# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
from mbrl.rl.rsl_rl import (
    RslRlSystemDynamicsCfg,
    RslRlNormalizerCfg,
    RslRlMbrlImaginationCfg,
    RslRlMbrlPpoAlgorithmCfg,
)


# fmt: off

# 96-dim state normalizer: [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
#                            joint_pos(29), joint_vel(29), joint_torque(29)]
# Placeholder values — must be refined from Init phase data collection.
_STATE_MEAN = [
    # base_lin_vel (3)
    0.0, 0.0, 0.0,
    # base_ang_vel (3)
    0.0, 0.0, 0.0,
    # projected_gravity (3)
    0.0, 0.0, -1.0,
    # joint_pos (29) — zeros (relative to default)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # joint_vel (29) — zeros (placeholder)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    # joint_torque (29) — zeros (placeholder)
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]

_STATE_STD = [
    # base_lin_vel (3)
    0.5, 0.5, 0.1,
    # base_ang_vel (3)
    0.3, 0.3, 0.5,
    # projected_gravity (3)
    0.02, 0.02, 0.04,
    # joint_pos (29) — placeholder
    0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
    0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
    0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
    # joint_vel (29) — placeholder
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
    # joint_torque (29) — placeholder
    15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
    15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
    15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0,
]

_ACTION_MEAN = [0.0] * 29
_ACTION_STD = [1.0] * 29

# fmt: on


@configclass
class G1FlatPPOPretrainRunnerCfg(G1FlatPPORunnerCfg):
    class_name: str = "MBPOOnPolicyRunner"

    system_dynamics = RslRlSystemDynamicsCfg(
        ensemble_size=5,
        history_horizon=32,
        architecture_config={
            "type": "rnn",
            "rnn_type": "gru",
            "rnn_num_layers": 2,
            "rnn_hidden_size": 512,
            "state_mean_shape": [256],
            "state_logstd_shape": [256],
            "extension_shape": [256],
            "contact_shape": [128],
            "termination_shape": [128],
        },
        freeze_auxiliary=False,
    )
    imagination = RslRlMbrlImaginationCfg(
        num_envs=0,
        num_steps_per_env=0,
        max_episode_length=0,
        command_resample_interval_range=None,
        uncertainty_penalty_weight=-0.0,
        state_normalizer=RslRlNormalizerCfg(
            mean=_STATE_MEAN,
            std=_STATE_STD,
        ),
        action_normalizer=RslRlNormalizerCfg(
            mean=_ACTION_MEAN,
            std=_ACTION_STD,
        ),
    )
    algorithm = RslRlMbrlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        policy_learning_rate=1.0e-3,
        system_dynamics_learning_rate=1.0e-3,
        system_dynamics_weight_decay=0.0,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        system_dynamics_forecast_horizon=8,
        system_dynamics_loss_weights={
            "state": 1.0,
            "sequence": 1.0,
            "bound": 1.0,
            "kl": 0.1,
            "extension": 1.0,
            "contact": 1.0,
            "termination": 1.0,
        },
        system_dynamics_num_mini_batches=20,
        system_dynamics_mini_batch_size=5000,
        system_dynamics_replay_buffer_size=1000,
        system_dynamics_num_eval_trajectories=100,
        system_dynamics_len_eval_trajectory=400,
        system_dynamics_eval_traj_noise_scale=[0.1, 0.2, 0.4, 0.5, 0.8],
    )
    run_name = "pretrain"
    load_system_dynamics = False
    system_dynamics_load_path = None
    system_dynamics_warmup_iterations = 0
    system_dynamics_num_visualizations = 4
    system_dynamics_state_idx_dict = {
        r"$v$\n$[m/s]$": [0, 1, 2],
        r"$\omega$\n$[rad/s]$": [3, 4, 5],
        r"$g$\n$[1]$": [6, 7, 8],
        r"$q$\n$[rad]$": list(range(9, 38)),
        r"$\dot{q}$\n$[rad/s]$": list(range(38, 67)),
        r"$\tau$\n$[Nm]$": list(range(67, 96)),
    }
    pca_obs_buf_size = 10000

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000


@configclass
class G1FlatPPOFinetuneRunnerCfg(G1FlatPPOPretrainRunnerCfg):
    resume = True
    load_run = ".*_pretrain"
    load_policy = False
    load_system_dynamics = True
    system_dynamics_load_path = None
    system_dynamics_warmup_iterations = 0
    run_name = "finetune"

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # override imagination
        self.imagination.num_envs = 8192
        self.imagination.num_steps_per_env = 24
        self.imagination.max_episode_length = 256
        self.imagination.command_resample_interval_range = [100, 120]
        self.imagination.uncertainty_penalty_weight = -0.0


@configclass
class G1FlatPPOVisualizeRunnerCfg(G1FlatPPOPretrainRunnerCfg):
    resume = True
    load_system_dynamics = True
    load_run = ".*_pretrain"
    load_checkpoint = "model_.*.pt"
    system_dynamics_load_path = None
    run_name = "visualize"
