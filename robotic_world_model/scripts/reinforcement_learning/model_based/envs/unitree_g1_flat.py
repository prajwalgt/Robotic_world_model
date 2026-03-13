from envs.base import BaseEnv
import torch
from tensordict import TensorDict


class UnitreeG1FlatEnv(BaseEnv):

    # Hip/knee joint indices within the 37-DOF joint vector for selective torque/acc penalties.
    # These must match the joint ordering from the G1 URDF used during data collection.
    # Default ordering (from G1_CFG actuator groups):
    #   legs: left_hip_yaw(0), right_hip_yaw(1), left_hip_roll(2), right_hip_roll(3),
    #         left_hip_pitch(4), right_hip_pitch(5), left_knee(6), right_knee(7), torso(8)
    #   feet: left_ankle_pitch(9), right_ankle_pitch(10), left_ankle_roll(11), right_ankle_roll(12)
    #   arms: shoulder/elbow/finger joints (13-36)
    # NOTE: Verify these indices match your data by checking joint_names from the Init phase.
    HIP_KNEE_INDICES = list(range(0, 8))  # hip_yaw/roll/pitch (6) + knee (2)
    HIP_YAW_ROLL_INDICES = list(range(0, 4))  # hip_yaw (2) + hip_roll (2)
    ARM_INDICES = list(range(13, 23))  # shoulder_pitch/roll/yaw (6) + elbow_pitch/roll (4)
    FINGER_INDICES = list(range(23, 37))  # 14 finger joints
    TORSO_INDICES = [8]  # torso_joint

    def _init_additional_imagination_attributes(self):
        # G1 is a biped: 2 feet
        self.last_air_time = torch.zeros(self.num_envs, 2, device=self.device)
        self.current_air_time = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_contact_time = torch.zeros(self.num_envs, 2, device=self.device)
        self.current_contact_time = torch.zeros(self.num_envs, 2, device=self.device)

    def _reset_additional_imagination_attributes(self, env_ids):
        self.last_air_time[env_ids] = 0.0
        self.current_air_time[env_ids] = 0.0
        self.last_contact_time[env_ids] = 0.0
        self.current_contact_time[env_ids] = 0.0

    def _init_imagination_command(self):
        r = torch.empty(self.num_envs, device=self.device)
        self.base_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.base_velocity[:, 0] = r.uniform_(-1.0, 1.0)
        self.base_velocity[:, 1] = r.uniform_(-1.0, 1.0)
        self.base_velocity[:, 2] = r.uniform_(-1.0, 1.0)
        self.is_standing_env[:] = r.uniform_(0.0, 1.0) <= 0.02
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.base_velocity[standing_env_ids, :] = 0.0

    def _reset_imagination_command(self, env_ids):
        r = torch.empty(len(env_ids), device=self.device)
        self.base_velocity[env_ids, 0] = r.uniform_(-1.0, 1.0)
        self.base_velocity[env_ids, 1] = r.uniform_(-1.0, 1.0)
        self.base_velocity[env_ids, 2] = r.uniform_(-1.0, 1.0)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= 0.02
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.base_velocity[standing_env_ids, :] = 0.0

    def get_imagination_observation(self, state_history, action_history):
        state_history_denormalized, action_history_denormalized = self.dataset.denormalize(state_history[:, -1], action_history[:, -1])
        # State layout (120-dim): [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
        #                          joint_pos(37), joint_vel(37), joint_torque(37)]
        obs_base_lin_vel = state_history_denormalized[:, 0:3]
        obs_base_ang_vel = state_history_denormalized[:, 3:6]
        obs_projected_gravity = state_history_denormalized[:, 6:9]
        obs_joint_pos = state_history_denormalized[:, 9:46]
        obs_joint_vel = state_history_denormalized[:, 46:83]
        self.obs_last_action = action_history_denormalized
        if self.observation_noise:
            obs_base_lin_vel += 2 * (torch.rand_like(obs_base_lin_vel) - 0.5) * 0.1
            obs_base_ang_vel += 2 * (torch.rand_like(obs_base_ang_vel) - 0.5) * 0.2
            obs_projected_gravity += 2 * (torch.rand_like(obs_projected_gravity) - 0.5) * 0.05
            obs_joint_pos += 2 * (torch.rand_like(obs_joint_pos) - 0.5) * 0.01
            obs_joint_vel += 2 * (torch.rand_like(obs_joint_vel) - 0.5) * 1.5
        # Observation (123-dim): [lin_vel(3), ang_vel(3), gravity(3), cmd(3), joint_pos(37), joint_vel(37), action(37)]
        obs = torch.cat([obs_base_lin_vel, obs_base_ang_vel, obs_projected_gravity, self.base_velocity, obs_joint_pos, obs_joint_vel, self.obs_last_action], dim=1)
        obs = TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)
        self.last_obs = obs
        return obs

    def _parse_imagination_states(self, imagination_states_denormalized):
        base_lin_vel = imagination_states_denormalized[:, 0:3]
        base_ang_vel = imagination_states_denormalized[:, 3:6]
        projected_gravity = imagination_states_denormalized[:, 6:9]
        joint_pos = imagination_states_denormalized[:, 9:46]
        joint_vel = imagination_states_denormalized[:, 46:83]
        joint_torque = imagination_states_denormalized[:, 83:120]

        parsed_imagination_states = {
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_torque": joint_torque,
        }
        return parsed_imagination_states

    def _parse_extensions(self, extensions):
        if extensions is None:
            return None
        parsed_extensions = {}
        return parsed_extensions

    def _parse_contacts(self, contacts):
        # G1: only 2 foot contacts (no thigh contacts)
        foot_contact = torch.sigmoid(contacts[:, 0:2]).round() if contacts is not None else None

        parsed_contacts = {
            "foot_contact": foot_contact,
        }
        return parsed_contacts

    def _parse_terminations(self, terminations):
        parsed_terminations = torch.sigmoid(terminations).squeeze(-1).round().bool() if terminations is not None else None
        return parsed_terminations

    def _compute_imagination_reward_terms(self, parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts):

        base_lin_vel = parsed_imagination_states["base_lin_vel"]
        base_ang_vel = parsed_imagination_states["base_ang_vel"]
        projected_gravity = parsed_imagination_states["projected_gravity"]
        joint_pos = parsed_imagination_states["joint_pos"]
        joint_vel = parsed_imagination_states["joint_vel"]
        joint_torque = parsed_imagination_states["joint_torque"]
        # joint_vel from last obs at indices [49:86] (3+3+3+3+37=49, then 37 joint_vel dims)
        joint_acc = (joint_vel - self.last_obs["policy"][:, 49:86]) / self._step_dt
        foot_contact = parsed_contacts["foot_contact"]

        # Velocity tracking rewards
        lin_vel_error = torch.sum(torch.square(self.base_velocity[:, :2] - base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.base_velocity[:, 2] - base_ang_vel[:, 2])

        track_lin_vel_xy_exp = torch.exp(-lin_vel_error / 0.25)
        track_ang_vel_z_exp = torch.exp(-ang_vel_error / 0.25)
        lin_vel_z_l2 = torch.square(base_lin_vel[:, 2])
        ang_vel_xy_l2 = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)

        # Torque and acceleration penalties (only on hip/knee joints)
        dof_torques_l2 = torch.sum(torch.square(joint_torque[:, self.HIP_KNEE_INDICES]), dim=1)
        dof_acc_l2 = torch.sum(torch.square(joint_acc[:, self.HIP_KNEE_INDICES]), dim=1)

        action_rate_l2 = torch.sum(torch.square(self.obs_last_action - rollout_action), dim=1)

        # Biped feet air time (threshold 0.4)
        if foot_contact is not None:
            first_contact = (self.current_contact_time > 0.0) * (self.current_contact_time < (self._step_dt + 1.0e-8))
            feet_air_time = torch.sum(torch.clamp(self.last_air_time - 0.4, min=0.0) * first_contact, dim=1) * (
                torch.norm(self.base_velocity[:, :2], dim=1) > 0.1
            )

            is_contact = foot_contact.bool()
            is_first_contact = (self.current_air_time > 0) * is_contact
            is_first_detached = (self.current_contact_time > 0) * ~is_contact
            self.last_air_time = torch.where(
                is_first_contact,
                self.current_air_time + self._step_dt,
                self.last_air_time,
            )
            self.current_air_time = torch.where(~is_contact, self.current_air_time + self._step_dt, 0.0)
            self.last_contact_time = torch.where(
                is_first_detached,
                self.current_contact_time + self._step_dt,
                self.last_contact_time,
            )
            self.current_contact_time = torch.where(is_contact, self.current_contact_time + self._step_dt, 0.0)
        else:
            feet_air_time = torch.zeros(self.num_envs, device=self.device)

        # G1 has no thigh contacts
        undesired_contacts = torch.zeros(self.num_envs, device=self.device)

        # Termination penalty
        termination_penalty = torch.zeros(self.num_envs, device=self.device)

        stand_still = torch.sum(torch.abs(joint_pos), dim=1) * (torch.norm(self.base_velocity, dim=1) < 0.05)
        flat_orientation_l2 = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

        # Joint deviation rewards (joint_pos is already relative to default)
        joint_deviation_hip = torch.sum(torch.abs(joint_pos[:, self.HIP_YAW_ROLL_INDICES]), dim=1)
        joint_deviation_arms = torch.sum(torch.abs(joint_pos[:, self.ARM_INDICES]), dim=1)
        joint_deviation_fingers = torch.sum(torch.abs(joint_pos[:, self.FINGER_INDICES]), dim=1)
        joint_deviation_torso = torch.sum(torch.abs(joint_pos[:, self.TORSO_INDICES]), dim=1)

        dof_pos_limits = torch.zeros(self.num_envs, device=self.device)

        self.imagination_reward_per_step = {
            "track_lin_vel_xy_exp": track_lin_vel_xy_exp,
            "track_ang_vel_z_exp": track_ang_vel_z_exp,
            "lin_vel_z_l2": lin_vel_z_l2,
            "ang_vel_xy_l2": ang_vel_xy_l2,
            "dof_torques_l2": dof_torques_l2,
            "dof_acc_l2": dof_acc_l2,
            "action_rate_l2": action_rate_l2,
            "feet_air_time": feet_air_time,
            "undesired_contacts": undesired_contacts,
            "termination_penalty": termination_penalty,
            "stand_still": stand_still,
            "flat_orientation_l2": flat_orientation_l2,
            "joint_deviation_hip": joint_deviation_hip,
            "joint_deviation_arms": joint_deviation_arms,
            "joint_deviation_fingers": joint_deviation_fingers,
            "joint_deviation_torso": joint_deviation_torso,
            "dof_pos_limits": dof_pos_limits,
        }
        last_obs = torch.cat([base_lin_vel, base_ang_vel, projected_gravity, self.base_velocity, joint_pos, joint_vel, rollout_action], dim=1)
        self.last_obs = TensorDict({"policy": last_obs}, batch_size=[self.num_envs], device=self.device)

    def _apply_interval_events(self, imagination_states_denormalized, parsed_imagination_states, event_ids):
        if len(event_ids) == 0:
            imagination_states, _ = self.dataset.normalize(imagination_states_denormalized, None)
        else:
            base_lin_vel = parsed_imagination_states["base_lin_vel"]
            velocity_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
            r = torch.empty(len(event_ids), device=self.device)
            base_lin_vel[event_ids, 0] += r.uniform_(*velocity_range["x"])
            base_lin_vel[event_ids, 1] += r.uniform_(*velocity_range["y"])
            imagination_states_denormalized[event_ids, 0:3] = base_lin_vel[event_ids, 0:3]
            imagination_states, _ = self.dataset.normalize(imagination_states_denormalized, None)
        return imagination_states

    @property
    def state_dim(self):
        return 120

    @property
    def observation_dim(self):
        return 123

    @property
    def action_dim(self):
        return 37
