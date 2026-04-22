from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg, G1Rewards
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg, RewardsCfg

from mbrl.mbrl.envs.mdp.commands import UniformVelocityCommand_Visualize, SampleUniformVelocityCommand
import mbrl.tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class G1RewardsCfg_TRAIN(G1Rewards):
    stand_still = RewTerm(
        func=mdp.joint_pos_stand_still, weight=-1.0, params={"command_name": "base_velocity", "threshold": 0.05}
    )


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):

    rewards: G1RewardsCfg_TRAIN = G1RewardsCfg_TRAIN()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards for flat terrain (from G1FlatEnvCfg in isaaclab_tasks)
        # self.rewards.track_ang_vel_z_exp.weight = 1.0
        # self.rewards.lin_vel_z_l2.weight = -0.2
        # self.rewards.action_rate_l2.weight = -0.005
        # self.rewards.dof_acc_l2.weight = -1.0e-7
        # self.rewards.feet_air_time.weight = 0.75
        # self.rewards.feet_air_time.params["threshold"] = 0.4
        # self.rewards.dof_torques_l2.weight = -2.0e-6
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        # )
        # self.rewards.flat_orientation_l2.weight = -5.0
        # change terrain to flat
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.feet_air_time.weight = 0.5
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.action_rate_l2.weight = -0.05
        # self.rewards.undesired_contacts.weight = -1.0

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class G1FlatEnvCfg_INIT(G1FlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # revert rewards to rough-terrain defaults for initial data collection
        # self.rewards.track_ang_vel_z_exp.weight = 2.0
        # self.rewards.lin_vel_z_l2.weight = 0.0
        # self.rewards.action_rate_l2.weight = -0.005
        # self.rewards.dof_acc_l2.weight = -1.25e-7
        # self.rewards.feet_air_time.weight = 0.25
        # self.rewards.dof_torques_l2.weight = -1.5e-7
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        # )
        # self.rewards.flat_orientation_l2.weight = -1.0


@configclass
class ObservationsCfg_PRETRAIN(ObservationsCfg):

    @configclass
    class SystemStateCfg(ObsGroup):

        # observation terms (order preserved)
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

        # observation terms (order preserved)
        pred_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemExtensionCfg(ObsGroup):

        pass

    @configclass
    class SystemContactCfg(ObsGroup):

        # G1 has no thigh contacts (undesired_contacts = None)
        # Only foot contacts on ankle roll links (2 feet)
        foot_contact = ObsTerm(
            func=mdp.body_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"), "threshold": 1.0},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SystemTerminationCfg(ObsGroup):

        # G1 uses torso_link as base body
        base_contact = ObsTerm(
            func=mdp.body_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link"), "threshold": 1.0},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    system_state: SystemStateCfg = SystemStateCfg()
    system_action: SystemActionCfg = SystemActionCfg()
    # system_extension: SystemExtensionCfg = SystemExtensionCfg()
    system_contact: SystemContactCfg = SystemContactCfg()
    system_termination: SystemTerminationCfg = SystemTerminationCfg()


@configclass
class G1FlatEnvCfg_PRETRAIN(G1FlatEnvCfg):

    # override observation terms
    observations: ObservationsCfg_PRETRAIN = ObservationsCfg_PRETRAIN()


@configclass
class G1FlatEnvCfg_FINETUNE(G1FlatEnvCfg_PRETRAIN):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # override commands
        self.commands.base_velocity.class_type = SampleUniformVelocityCommand


@configclass
class G1FlatEnvCfg_VISUALIZE(G1FlatEnvCfg_PRETRAIN):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for visualize
        self.scene.num_envs = 10
        self.scene.env_spacing = 2.5
        # disable randomization for visualize
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # override commands
        self.commands.base_velocity.class_type = UniformVelocityCommand_Visualize
        self.commands.base_velocity.resampling_time_range = (2.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # override randomization
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
