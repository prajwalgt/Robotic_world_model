from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from isaaclab_assets import G1_29DOF_CFG

from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg, G1Rewards
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as base_mdp

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

from mbrl.mbrl.envs.mdp.commands import UniformVelocityCommand_Visualize, SampleUniformVelocityCommand
import mbrl.tasks.manager_based.locomotion.velocity.mdp as mdp

# Joint names for the 29-DOF G1 (legs + waist + arms with wrists, no fingers)
G1_29DOF_JOINT_NAMES = [
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint",
    ".*_hip_pitch_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
    "waist_.*_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_.*_joint",
]


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

        # Switch to 29-DOF robot (legs + waist + arms with wrists, no fingers)
        self.scene.robot = G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.activate_contact_sensors = True

        # Restrict actions to the 29 actuated joints only
        self.actions.joint_pos = base_mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=G1_29DOF_JOINT_NAMES, scale=0.5, use_default_offset=True
        )

        # Fix reward terms that reference 37-DOF-specific joint names
        # torso_joint → replaced by waist joints in 29-DOF; upper body wqd = -1.0
        self.rewards.joint_deviation_torso = RewTerm(
            func=base_mdp.joint_deviation_l1,
            weight=-1.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*_joint"])},
        )
        # No finger joints in 29-DOF robot
        self.rewards.joint_deviation_fingers = None
        # elbow in 29-DOF is .*_elbow_joint (not elbow_pitch + elbow_roll); upper body wqd = -1.0
        self.rewards.joint_deviation_arms = RewTerm(
            func=base_mdp.joint_deviation_l1,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_shoulder_pitch_joint",
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        ".*_elbow_joint",
                        ".*_wrist_.*_joint",
                    ],
                )
            },
        )
        # lower body hip deviation wqd = -0.1
        self.rewards.joint_deviation_hip.weight = -0.1

        # Table S6 reward weights for G1
        self.rewards.track_ang_vel_z_exp.weight = 0.5  # wωz
        self.rewards.lin_vel_z_l2.weight = -2.0  # wvz
        self.rewards.action_rate_l2.weight = -0.05  # w˙a
        self.rewards.dof_acc_l2.weight = -2.5e-7  # w¨q, all 29 joints
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=G1_29DOF_JOINT_NAMES)
        self.rewards.feet_air_time.weight = 0.0  # wfa = 0 for G1
        self.rewards.dof_torques_l2.weight = -2.5e-5  # wqτ, all 29 joints
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=G1_29DOF_JOINT_NAMES)
        self.rewards.flat_orientation_l2.weight = -5.0  # wg
        # wc = -1.0: penalize undesired contacts on knee links (no thigh_link in G1 29-DOF)
        self.rewards.undesired_contacts = RewTerm(
            func=base_mdp.undesired_contacts,
            weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_knee_link"), "threshold": 1.0},
        )
        # wfc = 1.0: reward feet clearance during swing
        self.rewards.foot_clearance = RewTerm(
            func=mdp.foot_clearance,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
                "target_height": 0.1,
                "std": 0.25,
                "tanh_mult": 2.0,
            },
        )
        # change terrain to flat
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
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
        self.rewards.flat_orientation_l2.weight = -1.0
        # revert terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.terrain.terrain_generator.curriculum = False
        self.scene.terrain.terrain_generator.difficulty_range = (0.0, 0.0)
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["pyramid_stairs_inv"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].proportion = 1.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope"].proportion = 0.0
        self.scene.terrain.terrain_generator.sub_terrains["hf_pyramid_slope_inv"].proportion = 0.0


@configclass
class ObservationsCfg_PRETRAIN(ObservationsCfg):

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Override policy observations to restrict to 29-DOF joints."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_29DOF_JOINT_NAMES)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_29DOF_JOINT_NAMES)},
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
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_29DOF_JOINT_NAMES)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_29DOF_JOINT_NAMES)},
        )
        joint_torque = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=G1_29DOF_JOINT_NAMES)},
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

        # Paper Table S3: body_contact (26) + foot_height (2) + foot_velocity (2) = 30 dims
        # Matching paper's privileged information for G1
        body_contact = ObsTerm(
            func=mdp.body_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        "pelvis",
                        "waist_yaw_link",
                        "waist_roll_link",
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
                        ".*_elbow_link",
                        ".*_wrist_roll_link",
                        ".*_wrist_pitch_link",
                        ".*_wrist_yaw_link",
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
