from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.cassie.flat_env_cfg import (
    CassieFlatEnvCfg as IsaacLabCassieFlatEnvCfg,
)
from isaaclab_tasks.manager_based.locomotion.velocity.config.cassie.rough_env_cfg import CassieRewardsCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg

from mbrl.mbrl.envs.mdp.commands import SampleUniformVelocityCommand, UniformVelocityCommand_Visualize
import mbrl.tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class RewardsCfg_TRAIN(CassieRewardsCfg):
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*tarsus"), "threshold": 1.0},
    )
    foot_clearance = RewTerm(
        func=mdp.foot_clearance,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*toe"), "target_height": 0.2, "std": 0.05, "tanh_mult": 2.0},
    )
    joint_deviation = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


@configclass
class CassieFlatEnvCfg(IsaacLabCassieFlatEnvCfg):
    rewards: RewardsCfg_TRAIN = RewardsCfg_TRAIN()

    def __post_init__(self):
        super().__post_init__()

        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=0.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*tarsus"), "threshold": 1.0},
        )
        self.rewards.joint_deviation_hip = None
        self.rewards.joint_deviation_toes = None
        self.rewards.dof_pos_limits = None
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.feet_air_time.weight = 5.0


@configclass
class CassieFlatEnvCfg_INIT(CassieFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.feet_air_time.weight = 2.5


@configclass
class ObservationsCfg_PRETRAIN(ObservationsCfg):
    @configclass
    class SystemStateCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        joint_torque = ObsTerm(func=mdp.joint_effort)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemActionCfg(ObsGroup):
        pred_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemExtensionCfg(ObsGroup):
        ankle_contact = ObsTerm(
            func=mdp.body_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*tarsus"), "threshold": 1.0},
        )
        toe_height = ObsTerm(func=mdp.body_height_w, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*toe")})
        toe_planar_velocity = ObsTerm(
            func=mdp.body_lin_vel_w_norm,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*toe")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemContactCfg(ObsGroup):
        toe_contact = ObsTerm(
            func=mdp.body_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*toe"), "threshold": 1.0},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemTerminationCfg(ObsGroup):
        pelvis_contact = ObsTerm(
            func=mdp.body_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*pelvis"), "threshold": 1.0},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    system_state: SystemStateCfg = SystemStateCfg()
    system_action: SystemActionCfg = SystemActionCfg()
    system_extension: SystemExtensionCfg = SystemExtensionCfg()
    system_contact: SystemContactCfg = SystemContactCfg()
    system_termination: SystemTerminationCfg = SystemTerminationCfg()


@configclass
class CassieFlatEnvCfg_PRETRAIN(CassieFlatEnvCfg):
    observations: ObservationsCfg_PRETRAIN = ObservationsCfg_PRETRAIN()


@configclass
class CassieFlatEnvCfg_FINETUNE(CassieFlatEnvCfg_PRETRAIN):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.class_type = SampleUniformVelocityCommand


@configclass
class CassieFlatEnvCfg_VISUALIZE(CassieFlatEnvCfg_PRETRAIN):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        self.commands.base_velocity.class_type = UniformVelocityCommand_Visualize
        self.commands.base_velocity.resampling_time_range = (2.0, 2.0)
        self.events.reset_base.func = mdp.reset_root_state_uniform_visualize
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "yaw": (1.57, 1.57)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.func = mdp.reset_joints_by_scale_visualize
