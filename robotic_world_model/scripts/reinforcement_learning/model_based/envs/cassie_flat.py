from envs.base import BaseEnv
import torch
from tensordict import TensorDict


class CassieFlatEnv(BaseEnv):
    FOOT_CLEARANCE_TARGET_HEIGHT = 0.2
    FOOT_CLEARANCE_STD = 0.05
    FOOT_CLEARANCE_TANH_MULT = 2.0

    def _init_additional_imagination_attributes(self):
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
        obs_base_lin_vel = state_history_denormalized[:, 0:3]
        obs_base_ang_vel = state_history_denormalized[:, 3:6]
        obs_projected_gravity = state_history_denormalized[:, 6:9]
        obs_joint_pos = state_history_denormalized[:, 9:21]
        obs_joint_vel = state_history_denormalized[:, 21:33]
        self.obs_last_action = action_history_denormalized
        if self.observation_noise:
            obs_base_lin_vel += 2 * (torch.rand_like(obs_base_lin_vel) - 0.5) * 0.1
            obs_base_ang_vel += 2 * (torch.rand_like(obs_base_ang_vel) - 0.5) * 0.2
            obs_projected_gravity += 2 * (torch.rand_like(obs_projected_gravity) - 0.5) * 0.05
            obs_joint_pos += 2 * (torch.rand_like(obs_joint_pos) - 0.5) * 0.01
            obs_joint_vel += 2 * (torch.rand_like(obs_joint_vel) - 0.5) * 1.5
        obs = torch.cat([obs_base_lin_vel, obs_base_ang_vel, obs_projected_gravity, self.base_velocity, obs_joint_pos, obs_joint_vel, self.obs_last_action], dim=1)
        obs = TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)
        self.last_obs = obs
        return obs

    def _parse_imagination_states(self, imagination_states_denormalized):
        return {
            "base_lin_vel": imagination_states_denormalized[:, 0:3],
            "base_ang_vel": imagination_states_denormalized[:, 3:6],
            "projected_gravity": imagination_states_denormalized[:, 6:9],
            "joint_pos": imagination_states_denormalized[:, 9:21],
            "joint_vel": imagination_states_denormalized[:, 21:33],
            "joint_torque": imagination_states_denormalized[:, 33:45],
        }

    def _parse_extensions(self, extensions):
        if extensions is None:
            return {}
        return {
            "ankle_contact": torch.sigmoid(extensions[:, 0:2]).round(),
            "toe_height": extensions[:, 2:4],
            "toe_planar_velocity": extensions[:, 4:6],
        }

    def _parse_contacts(self, contacts):
        if contacts is None:
            return {}
        return {"toe_contact": torch.sigmoid(contacts[:, 0:2]).round()}

    def _parse_terminations(self, terminations):
        return torch.sigmoid(terminations).squeeze(-1).round().bool() if terminations is not None else None

    def _compute_imagination_reward_terms(self, parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts):
        base_lin_vel = parsed_imagination_states["base_lin_vel"]
        base_ang_vel = parsed_imagination_states["base_ang_vel"]
        projected_gravity = parsed_imagination_states["projected_gravity"]
        joint_pos = parsed_imagination_states["joint_pos"]
        joint_vel = parsed_imagination_states["joint_vel"]
        joint_torque = parsed_imagination_states["joint_torque"]
        joint_acc = (joint_vel - self.last_obs["policy"][:, 12:24]) / self._step_dt

        toe_contact = parsed_contacts.get("toe_contact")
        ankle_contact = parsed_extensions.get("ankle_contact")
        toe_height = parsed_extensions.get("toe_height")
        toe_planar_velocity = parsed_extensions.get("toe_planar_velocity")

        lin_vel_error = torch.sum(torch.square(self.base_velocity[:, :2] - base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.base_velocity[:, 2] - base_ang_vel[:, 2])

        track_lin_vel_xy_exp = torch.exp(-lin_vel_error / 0.25)
        track_ang_vel_z_exp = torch.exp(-ang_vel_error / 0.25)
        lin_vel_z_l2 = torch.square(base_lin_vel[:, 2])
        ang_vel_xy_l2 = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)
        dof_torques_l2 = torch.sum(torch.square(joint_torque), dim=1)
        dof_acc_l2 = torch.sum(torch.square(joint_acc), dim=1)
        action_rate_l2 = torch.sum(torch.square(self.obs_last_action - rollout_action), dim=1)

        if toe_contact is not None:
            first_contact = (self.current_contact_time > 0.0) * (self.current_contact_time < (self._step_dt + 1.0e-8))
            feet_air_time = torch.sum((self.last_air_time - 0.3) * first_contact, dim=1) * (torch.norm(self.base_velocity[:, :2], dim=1) > 0.1)

            is_contact = toe_contact.bool()
            is_first_contact = (self.current_air_time > 0) * is_contact
            is_first_detached = (self.current_contact_time > 0) * ~is_contact
            self.last_air_time = torch.where(is_first_contact, self.current_air_time + self._step_dt, self.last_air_time)
            self.current_air_time = torch.where(~is_contact, self.current_air_time + self._step_dt, 0.0)
            self.last_contact_time = torch.where(is_first_detached, self.current_contact_time + self._step_dt, self.last_contact_time)
            self.current_contact_time = torch.where(is_contact, self.current_contact_time + self._step_dt, 0.0)
        else:
            feet_air_time = torch.zeros(self.num_envs, device=self.device)

        undesired_contacts = torch.sum(ankle_contact, dim=1) if ankle_contact is not None else torch.zeros(self.num_envs, device=self.device)
        flat_orientation_l2 = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
        joint_deviation = torch.sum(torch.abs(joint_pos), dim=1)
        termination_penalty = self.termination_flags.float() if self.termination_flags is not None else torch.zeros(self.num_envs, device=self.device)

        if toe_height is not None and toe_planar_velocity is not None:
            toe_height_error = torch.square(toe_height - self.FOOT_CLEARANCE_TARGET_HEIGHT)
            toe_velocity_tanh = torch.tanh(self.FOOT_CLEARANCE_TANH_MULT * toe_planar_velocity)
            foot_clearance = torch.exp(-torch.sum(toe_height_error * toe_velocity_tanh, dim=1) / self.FOOT_CLEARANCE_STD)
        else:
            foot_clearance = torch.zeros(self.num_envs, device=self.device)

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
            "flat_orientation_l2": flat_orientation_l2,
            "foot_clearance": foot_clearance,
            "joint_deviation": joint_deviation,
            "termination_penalty": termination_penalty,
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
        return 45

    @property
    def observation_dim(self):
        return 48

    @property
    def action_dim(self):
        return 12
