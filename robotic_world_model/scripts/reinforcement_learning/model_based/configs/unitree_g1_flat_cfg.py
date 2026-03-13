from .base_cfg import BaseConfig
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class UnitreeG1FlatConfig(BaseConfig):
    experiment_name: str = "offline"

    @dataclass
    class ExperimentConfig(BaseConfig.ExperimentConfig):
        environment: str = "unitree_g1_flat"

    @dataclass
    class EnvironmentConfig(BaseConfig.EnvironmentConfig):
        reward_term_weights: Dict[str, float] = field(default_factory=lambda: {
            "track_lin_vel_xy_exp": 1.0,
            "track_ang_vel_z_exp": 1.0,
            "lin_vel_z_l2": -0.2,
            "ang_vel_xy_l2": -0.05,
            "dof_torques_l2": -2.0e-6,
            "dof_acc_l2": -1.0e-7,
            "action_rate_l2": -0.005,
            "feet_air_time": 0.75,
            "undesired_contacts": 0.0,
            "termination_penalty": -200.0,
            "stand_still": -1.0,
            "flat_orientation_l2": -5.0,
            "joint_deviation_hip": -0.1,
            "joint_deviation_arms": -0.1,
            "joint_deviation_fingers": -0.05,
            "joint_deviation_torso": -0.1,
            "dof_pos_limits": 0.0,
        })
        uncertainty_penalty_weight: float = -1.0
        command_resample_interval_range: List[int] | None = field(default_factory=lambda: [100, 120])
        event_interval_range: List[int] = field(default_factory=lambda: [48, 96])

    @dataclass
    class DataConfig(BaseConfig.DataConfig):
        dataset_root: str = "assets"
        dataset_folder: str = "data"
        batch_data_size: int = 10000
        state_idx_dict: Dict[str, List[int]] = field(default_factory=lambda: {
            r"$v$\n$[m/s]$": [0, 1, 2],
            r"$\omega$\n$[rad/s]$": [3, 4, 5],
            r"$g$\n$[1]$": [6, 7, 8],
            r"$q$\n$[rad]$": list(range(9, 46)),
            r"$\dot{q}$\n$[rad/s]$": list(range(46, 83)),
            r"$\tau$\n$[Nm]$": list(range(83, 120)),
        })
        # 120-dim state normalizer — placeholder values, refine after Init phase data collection
        state_data_mean: List[float] = field(default_factory=lambda: [
            # base_lin_vel (3)
            0.0, 0.0, 0.0,
            # base_ang_vel (3)
            0.0, 0.0, 0.0,
            # projected_gravity (3)
            0.0, 0.0, -1.0,
            # joint_pos (37)
            *([0.0] * 37),
            # joint_vel (37)
            *([0.0] * 37),
            # joint_torque (37)
            *([0.0] * 37),
        ])
        state_data_std: List[float] = field(default_factory=lambda: [
            # base_lin_vel (3)
            0.5, 0.5, 0.1,
            # base_ang_vel (3)
            0.3, 0.3, 0.5,
            # projected_gravity (3)
            0.02, 0.02, 0.04,
            # joint_pos (37)
            *([0.15] * 37),
            # joint_vel (37)
            *([1.5] * 37),
            # joint_torque (37)
            *([15.0] * 37),
        ])
        action_data_mean: List[float] = field(default_factory=lambda: [0.0] * 37)
        action_data_std: List[float] = field(default_factory=lambda: [1.0] * 37)

    @dataclass
    class ModelArchitectureConfig(BaseConfig.ModelArchitectureConfig):
        history_horizon: int = 32
        forecast_horizon: int = 8
        ensemble_size: int = 5
        contact_dim: int = 2
        termination_dim: int = 1
        architecture_config: Dict[str, object] = field(default_factory=lambda: {
            "type": "rnn",
            "rnn_type": "gru",
            "rnn_num_layers": 2,
            "rnn_hidden_size": 512,
            "state_mean_shape": [256],
            "state_logstd_shape": [256],
            "extension_shape": [256],
            "contact_shape": [128],
            "termination_shape": [128],
        })
        resume_path: str | None = None

    @dataclass
    class PolicyArchitectureConfig(BaseConfig.PolicyArchitectureConfig):
        observation_dim: int = 123
        action_dim: int = 37
        resume_path: str | None = None

    @dataclass
    class PolicyAlgorithmConfig(BaseConfig.PolicyAlgorithmConfig):
        learning_rate: float = 1.0e-4
        entropy_coef: float = 0.0001

    @dataclass
    class PolicyTrainingConfig(BaseConfig.PolicyTrainingConfig):
        save_interval: int = 50
        max_iterations: int = 500

    experiment_config: ExperimentConfig = field(default_factory=ExperimentConfig)
    environment_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    model_architecture_config: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    policy_architecture_config: PolicyArchitectureConfig = field(default_factory=PolicyArchitectureConfig)
    policy_algorithm_config: PolicyAlgorithmConfig = field(default_factory=PolicyAlgorithmConfig)
    policy_training_config: PolicyTrainingConfig = field(default_factory=PolicyTrainingConfig)
