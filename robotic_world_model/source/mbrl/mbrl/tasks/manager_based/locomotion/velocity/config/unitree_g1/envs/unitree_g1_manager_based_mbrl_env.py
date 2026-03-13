# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import torch
from tensordict import TensorDict

from mbrl.mbrl.envs import ManagerBasedMBRLEnv


class UnitreeG1ManagerBasedMBRLEnv(ManagerBasedMBRLEnv):

    # G1 joint indices (0-indexed within the 37-DOF joint vector) for selective reward computation.
    # These are resolved at init time from the robot's joint_names to avoid hardcoding.
    _hip_knee_indices = None  # for dof_torques_l2 and dof_acc_l2
    _hip_yaw_roll_indices = None  # for joint_deviation_hip
    _arm_indices = None  # for joint_deviation_arms
    _finger_indices = None  # for joint_deviation_fingers
    _torso_indices = None  # for joint_deviation_torso

    def _init_additional_attributes(self):
        self.default_joint_pos = self.scene["robot"].data.default_joint_pos[0]
        self.default_joint_vel = self.scene["robot"].data.default_joint_vel[0]
        self.base_velocity = None

        # Resolve joint indices by name from the articulation
        joint_names = self.scene["robot"].data.joint_names
        self._hip_knee_indices = self._resolve_joint_indices(joint_names, [".*_hip_.*", ".*_knee_joint"])
        self._hip_yaw_roll_indices = self._resolve_joint_indices(joint_names, [".*_hip_yaw_joint", ".*_hip_roll_joint"])
        self._arm_indices = self._resolve_joint_indices(
            joint_names,
            [
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
            ],
        )
        self._finger_indices = self._resolve_joint_indices(
            joint_names,
            [
                ".*_five_joint",
                ".*_three_joint",
                ".*_six_joint",
                ".*_four_joint",
                ".*_zero_joint",
                ".*_one_joint",
                ".*_two_joint",
            ],
        )
        self._torso_indices = self._resolve_joint_indices(joint_names, ["torso_joint"])

    @staticmethod
    def _resolve_joint_indices(joint_names, patterns):
        """Find indices of joints matching regex patterns."""
        import re

        indices = []
        for i, name in enumerate(joint_names):
            for pattern in patterns:
                if re.fullmatch(pattern, name):
                    indices.append(i)
                    break
        return indices

    def _init_additional_imagination_attributes(self):
        # G1 is a biped: 2 feet (left and right ankle_roll_link)
        self.last_air_time = torch.zeros(self.num_imagination_envs, 2, device=self.device)
        self.current_air_time = torch.zeros(self.num_imagination_envs, 2, device=self.device)
        self.last_contact_time = torch.zeros(self.num_imagination_envs, 2, device=self.device)
        self.current_contact_time = torch.zeros(self.num_imagination_envs, 2, device=self.device)

    def _reset_additional_imagination_attributes(self, env_ids):
        self.last_air_time[env_ids] = 0.0
        self.current_air_time[env_ids] = 0.0
        self.last_contact_time[env_ids] = 0.0
        self.current_contact_time[env_ids] = 0.0

    def get_imagination_observation(self, state_history, action_history, observation_noise=True):
        # State layout (120-dim): [base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
        #                          joint_pos(37), joint_vel(37), joint_torque(37)]
        state = self.imagination_state_normalizer.inverse(state_history[:, -1])
        obs_base_lin_vel = state[:, 0:3]
        obs_base_ang_vel = state[:, 3:6]
        obs_projected_gravity = state[:, 6:9]
        obs_joint_pos = state[:, 9:46]
        obs_joint_vel = state[:, 46:83]
        self.obs_last_action = self.imagination_action_normalizer.inverse(action_history[:, -1])
        if observation_noise:
            obs_base_lin_vel = obs_base_lin_vel + 2 * (torch.rand_like(obs_base_lin_vel) - 0.5) * 0.1
            obs_base_ang_vel = obs_base_ang_vel + 2 * (torch.rand_like(obs_base_ang_vel) - 0.5) * 0.2
            obs_projected_gravity = obs_projected_gravity + 2 * (torch.rand_like(obs_projected_gravity) - 0.5) * 0.05
            obs_joint_pos = obs_joint_pos + 2 * (torch.rand_like(obs_joint_pos) - 0.5) * 0.01
            obs_joint_vel = obs_joint_vel + 2 * (torch.rand_like(obs_joint_vel) - 0.5) * 1.5
        # Observation (123-dim): [lin_vel(3), ang_vel(3), gravity(3), cmd(3), joint_pos(37), joint_vel(37), action(37)]
        obs = torch.cat(
            [obs_base_lin_vel, obs_base_ang_vel, obs_projected_gravity, self.base_velocity, obs_joint_pos, obs_joint_vel, self.obs_last_action],
            dim=1,
        )
        obs = TensorDict({"policy": obs}, batch_size=[self.num_imagination_envs], device=self.device)
        self.last_obs = obs
        return obs

    def _parse_imagination_states(self, imagination_states_denormalized):
        # 120-dim state: [0:3] lin_vel, [3:6] ang_vel, [6:9] gravity, [9:46] joint_pos, [46:83] joint_vel, [83:120] joint_torque
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
        # joint_vel from last obs is at indices [49:86] (3+3+3+3+37=49, then 37 joint_vel dims)
        joint_acc = (joint_vel - self.last_obs["policy"][:, 49:86]) / self.step_dt
        foot_contact = parsed_contacts["foot_contact"]

        # Velocity tracking rewards
        lin_vel_error = torch.sum(torch.square(self.base_velocity[:, :2] - base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.base_velocity[:, 2] - base_ang_vel[:, 2])

        track_lin_vel_xy_exp = torch.exp(-lin_vel_error / 0.25)
        track_ang_vel_z_exp = torch.exp(-ang_vel_error / 0.25)
        lin_vel_z_l2 = torch.square(base_lin_vel[:, 2])
        ang_vel_xy_l2 = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)

        # Torque and acceleration penalties (only on hip/knee joints)
        dof_torques_l2 = torch.sum(torch.square(joint_torque[:, self._hip_knee_indices]), dim=1)
        dof_acc_l2 = torch.sum(torch.square(joint_acc[:, self._hip_knee_indices]), dim=1)

        action_rate_l2 = torch.sum(torch.square(self.obs_last_action - rollout_action), dim=1)

        # Biped feet air time (threshold 0.4 instead of 0.5 for quadruped)
        if foot_contact is not None:
            first_contact = (self.current_contact_time > 0.0) * (self.current_contact_time < (self.step_dt + 1.0e-8))
            feet_air_time = torch.sum(torch.clamp(self.last_air_time - 0.4, min=0.0) * first_contact, dim=1) * (
                torch.norm(self.base_velocity[:, :2], dim=1) > 0.1
            )

            is_contact = foot_contact.bool()
            is_first_contact = (self.current_air_time > 0) * is_contact
            is_first_detached = (self.current_contact_time > 0) * ~is_contact
            self.last_air_time = torch.where(
                is_first_contact,
                self.current_air_time + self.step_dt,
                self.last_air_time,
            )
            self.current_air_time = torch.where(~is_contact, self.current_air_time + self.step_dt, 0.0)
            self.last_contact_time = torch.where(
                is_first_detached,
                self.current_contact_time + self.step_dt,
                self.last_contact_time,
            )
            self.current_contact_time = torch.where(is_contact, self.current_contact_time + self.step_dt, 0.0)
        else:
            feet_air_time = torch.zeros(self.num_imagination_envs, device=self.device)

        # G1 has no thigh contacts — undesired_contacts is always zero
        undesired_contacts = torch.zeros(self.num_imagination_envs, device=self.device)

        # Termination penalty (based on termination flag from dynamics model)
        termination_penalty = torch.zeros(self.num_imagination_envs, device=self.device)

        stand_still = torch.sum(torch.abs(joint_pos), dim=1) * (torch.norm(self.base_velocity, dim=1) < 0.05)
        flat_orientation_l2 = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)

        # Joint deviation rewards (joint_pos is already relative to default)
        joint_deviation_hip = torch.sum(torch.abs(joint_pos[:, self._hip_yaw_roll_indices]), dim=1)
        joint_deviation_arms = torch.sum(torch.abs(joint_pos[:, self._arm_indices]), dim=1)
        joint_deviation_fingers = torch.sum(torch.abs(joint_pos[:, self._finger_indices]), dim=1)
        joint_deviation_torso = torch.sum(torch.abs(joint_pos[:, self._torso_indices]), dim=1)

        # dof_pos_limits: placeholder (would need joint limit info to compute properly)
        dof_pos_limits = torch.zeros(self.num_imagination_envs, device=self.device)

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
            "feet_slide": torch.zeros(self.num_imagination_envs,device=self.device),
            "termination_penalty": termination_penalty,
            "stand_still": stand_still,
            "flat_orientation_l2": flat_orientation_l2,
            "joint_deviation_hip": joint_deviation_hip,
            "joint_deviation_arms": joint_deviation_arms,
            "joint_deviation_fingers": joint_deviation_fingers,
            "joint_deviation_torso": joint_deviation_torso,
            "dof_pos_limits": dof_pos_limits,
        }
        # Update last_obs for next step (123-dim)
        last_obs = torch.cat(
            [base_lin_vel, base_ang_vel, projected_gravity, self.base_velocity, joint_pos, joint_vel, rollout_action],
            dim=1,
        )
        self.last_obs = TensorDict({"policy": last_obs}, batch_size=[self.num_imagination_envs], device=self.device)
