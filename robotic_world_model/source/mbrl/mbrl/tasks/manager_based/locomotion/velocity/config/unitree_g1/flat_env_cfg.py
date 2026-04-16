from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg as IsaacLabG1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg

from mbrl.mbrl.envs.mdp.commands import UniformVelocityCommand_Visualize, SampleUniformVelocityCommand
import mbrl.tasks.manager_based.locomotion.velocity.mdp as mdp

@configclass
class G1FlatEnvCfg(IsaacLabG1FlatEnvCfg):
    pass


@configclass
class G1FlatEnvCfg_INIT(G1FlatEnvCfg):
    pass


@configclass
class ObservationsCfg_PRETRAIN(ObservationsCfg):

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Use the full 37-DOF policy observation layout."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)

    @configclass
    class SystemStateCfg(ObsGroup):

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
        )
        joint_torque = ObsTerm(
            func=mdp.joint_effort,
        )

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

        # 30 privileged contact dims for the 37-DOF G1:
        # pelvis + torso + bilateral hip/leg/arm links + a subset of finger links.
        body_contact = ObsTerm(
            func=mdp.body_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        "pelvis",
                        "torso_link",
                        ".*_hip_yaw_link",
                        ".*_hip_roll_link",
                        ".*_hip_pitch_link",
                        ".*_knee_link",
                        ".*_ankle_pitch_link",
                        ".*_ankle_roll_link",
                        ".*_shoulder_pitch_link",
                        ".*_shoulder_roll_link",
                        ".*_shoulder_yaw_link",
                        ".*_elbow_pitch_link",
                        ".*_elbow_roll_link",
                        ".*_five_link",
                        ".*_three_link",
                        ".*_six_link",
                    ],
                ),
                "threshold": 1.0,
            },
        )
        foot_height = ObsTerm(
            func=mdp.body_height_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
        )
        foot_velocity = ObsTerm(
            func=mdp.body_lin_vel_w_norm,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
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
    policy: PolicyCfg = PolicyCfg()
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
