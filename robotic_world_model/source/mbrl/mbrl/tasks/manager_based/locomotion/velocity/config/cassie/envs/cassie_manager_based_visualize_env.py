from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply

from mbrl.mbrl.envs import ManagerBasedVisualizeEnv
from mbrl.mbrl.envs.mdp.events import reset_joints_to_specified, reset_root_state_to_specified

from .cassie_manager_based_mbrl_env import CassieManagerBasedMBRLEnv


class CassieManagerBasedVisualizeEnv(ManagerBasedVisualizeEnv, CassieManagerBasedMBRLEnv):
    def _reset_imagination_sim(self, parsed_imagination_states):
        base_lin_vel = parsed_imagination_states["base_lin_vel"]
        base_ang_vel = parsed_imagination_states["base_ang_vel"]
        joint_pos = parsed_imagination_states["joint_pos"] + self.default_joint_pos
        joint_vel = parsed_imagination_states["joint_vel"] + self.default_joint_vel

        root_pos_w = self.scene["robot"].data.root_pos_w[self.env_ids_real] - self.scene.env_origins[self.env_ids_real]
        root_pos_w = root_pos_w + self.scene.env_origins[self.env_ids_imagination]
        root_quat_w = self.scene["robot"].data.root_quat_w[self.env_ids_real]
        base_lin_vel_w = quat_apply(root_quat_w, base_lin_vel)
        base_ang_vel_w = quat_apply(root_quat_w, base_ang_vel)
        velocities = torch.cat([base_lin_vel_w, base_ang_vel_w], dim=1)

        reset_joints_to_specified(self, self.env_ids_imagination, joint_pos, joint_vel)
        reset_root_state_to_specified(self, self.env_ids_imagination, root_pos_w, root_quat_w, velocities)
