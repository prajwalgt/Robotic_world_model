Step 1: Create directory structure

Mirror the anymal_d/ folder:

source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/unitree_g1/
├── __init__.py
├── flat_env_cfg.py
├── agents/
│   ├── __init__.py
│   └── rsl_rl_ppo_cfg.py
└── envs/
    ├── unitree_g1_manager_based_mbrl_env.py
    └── unitree_g1_manager_based_visualize_env.py

Also create offline training files:
scripts/reinforcement_learning/model_based/envs/unitree_g1_flat.py
scripts/reinforcement_learning/model_based/configs/unitree_g1_flat_cfg.py

---
Step 2: flat_env_cfg.py — Environment configuration

Template:
source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/flat_env_cfg.py

Key changes:
- Inherit from G1RoughEnvCfg (from isaaclab_tasks...config.g1.rough_env_cfg) instead of
AnymalDRoughEnvCfg
- Inherit rewards from G1Rewards instead of the anymal rewards class
- Add stand_still reward term (using existing mdp.joint_pos_stand_still)
- Update SystemContactCfg:
  - Remove thigh_contact (G1 sets undesired_contacts = None)
  - foot_contact body names: ".*_ankle_roll_link" (not ".*FOOT")
- Update SystemTerminationCfg: body name "torso_link" (not "base")
- Reward weights from G1's flat env config (different from ANYmal D):
  - track_ang_vel_z_exp: 1.0, lin_vel_z_l2: -0.2, feet_air_time: 0.75
  - dof_torques_l2: -2.0e-6 (on hip/knee joints only), dof_acc_l2: -1.0e-7 (hip/knee
only)
  - flat_orientation_l2: -5.0, action_rate_l2: -0.005
- G1 uses feet_air_time_positive_biped reward (not feet_air_time)
- G1 has additional reward terms: termination_penalty, feet_slide,
joint_deviation_hip/arms/fingers/torso, dof_pos_limits
- Define 4 variant configs: _INIT, _PRETRAIN, _FINETUNE, _VISUALIZE

---
Step 3: unitree_g1_manager_based_mbrl_env.py — Imagination environment

Template: source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/envs/
anymal_d_manager_based_mbrl_env.py

Key changes:
- Air time buffers: shape (num_envs, 2) not (num_envs, 4) — 2 feet instead of 4
- _parse_imagination_states(): Update slice indices for 37-DOF:
  - joint_pos = states[:, 9:46], joint_vel = states[:, 46:83], joint_torque = states[:,
83:120]
- _parse_contacts(): Only foot_contact = sigmoid(contacts[:, 0:2]), no thigh contact
- get_imagination_observation(): Update slices for 37 joints
- _compute_imagination_reward_terms(): Major rewrite:
  - joint_acc: extract joint_vel from obs at indices [49:86] (after 3+3+3+3+37=49)
  - feet_air_time: biped version with threshold 0.4, operates on 2 feet
  - Remove undesired_contacts (set to zero)
  - Add termination_penalty, joint_deviation_hip/arms/fingers/torso (computed from
joint_pos since it's already relative to default)
  - dof_torques_l2 and dof_acc_l2 should sum only over hip/knee joint indices (need to
identify which of the 37 indices correspond to these joints — determine at runtime or
hardcode based on joint ordering)
  - Skip feet_slide in imagination (requires foot velocity not available in state)
  - Update last_obs construction for 123-dim observation

Important: The joint index mapping for selective reward computation (hip/knee only for
torque/acc penalties) needs to be determined. Options:
- Query self.scene["robot"].data.joint_names at init time
- Or hardcode based on the G1 URDF joint ordering

---
Step 4: unitree_g1_manager_based_visualize_env.py — Visualization env

Template: source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/envs/
anymal_d_manager_based_visualize_env.py

Minimal changes — inherits from both ManagerBasedVisualizeEnv and
UnitreeG1ManagerBasedMBRLEnv. The _reset_imagination_sim method works generically with
default_joint_pos/default_joint_vel from the articulation data.

---
Step 5: agents/rsl_rl_ppo_cfg.py — Training configs

Template: source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/agent
s/rsl_rl_ppo_cfg.py

Key changes:
- Inherit from G1FlatPPORunnerCfg (from
isaaclab_tasks...config.g1.agents.rsl_rl_ppo_cfg)
- System dynamics: Consider increasing rnn_hidden_size from 256 to 384 or 512 (120-dim
state vs 45-dim)
- State normalizer: 120-element mean/std vectors:
  - mean: [0,0,0, 0,0,0, 0,0,-1, <37 zeros for joint_pos>, <37 zeros for joint_vel>, <37
torque means - need empirical data>]
  - std: [0.5,0.5,0.1, 0.3,0.3,0.5, 0.02,0.02,0.04, <37 joint_pos stds>, <37 joint_vel
stds>, <37 torque stds>]
  - These MUST be computed from G1 simulation data. Use placeholder values initially,
then refine after running Init phase.
- Action normalizer: 37-element mean(0) and std(1)
- Update system_dynamics_state_idx_dict for 120-dim state with ranges for 37 joints
- Update imagination config: contact_dim=2
- Define 3 runner configs: Pretrain, Finetune, Visualize

---
Step 6: __init__.py — Environment registration

Template:
source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/__init__.py

Register 4 gym environments:
- Template-Isaac-Velocity-Flat-G1-Init-v0 → ManagerBasedRLEnv
- Template-Isaac-Velocity-Flat-G1-Pretrain-v0 → ManagerBasedRLEnv
- Template-Isaac-Velocity-Flat-G1-Finetune-v0 → UnitreeG1ManagerBasedMBRLEnv
- Template-Isaac-Velocity-Flat-G1-Visualize-v0 → UnitreeG1ManagerBasedVisualizeEnv

---
Step 7: Offline training files

scripts/reinforcement_learning/model_based/envs/unitree_g1_flat.py

Template: scripts/reinforcement_learning/model_based/envs/anymal_d_flat.py

- state_dim = 120, observation_dim = 123, action_dim = 37
- Air time buffers: 2 feet
- _parse_contacts(): 2 foot contacts from contacts[:, 0:2], no thigh
- Update all slice indices for 37-DOF joints
- Reward computation matching the imagination env (Step 3)
- Register in envs/__init__.py

scripts/reinforcement_learning/model_based/configs/unitree_g1_flat_cfg.py

Template: scripts/reinforcement_learning/model_based/configs/anymal_d_flat_cfg.py

- contact_dim = 2, termination_dim = 1
- Updated reward_term_weights matching G1 rewards
- 120-dim state_data_mean/state_data_std (placeholder values initially)
- 37-dim action_data_mean/action_data_std
- state_idx_dict with ranges for 37 joints
- observation_dim = 123, action_dim = 37
- Register in configs/__init__.py

---
Step 8: Update imports/registrations

- scripts/reinforcement_learning/model_based/envs/__init__.py — add unitree_g1_flat
mapping
- scripts/reinforcement_learning/model_based/configs/__init__.py — export
UnitreeG1FlatConfig
- source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/__init__.py — import
unitree_g1 if needed

---
Key decisions

1. All 37 joints are used (not just leg joints) — the isaaclab_tasks action space uses
joint_names=[".*"] and G1 rewards penalize arm/finger/torso deviation
2. Contact dim = 2 (feet only) — G1 has no thigh contact penalties (undesired_contacts =
None)
3. Skip feet_slide in imagination — requires foot velocity not available in predicted
states
4. Normalizer stats are placeholders until data is collected from the Init phase
5. Larger RNN may be needed (384-512 hidden) for the 120-dim state vs ANYmal's 45-dim

---
Files to modify (existing)

┌────────────────────────────────────────────────────────────────┬───────────────────┐
│                              File                              │      Change       │
├────────────────────────────────────────────────────────────────┼───────────────────┤
│ scripts/reinforcement_learning/model_based/envs/__init__.py    │ Add G1 env import │
├────────────────────────────────────────────────────────────────┼───────────────────┤
│ scripts/reinforcement_learning/model_based/configs/__init__.py │ Add G1 config     │
│                                                                │ export            │
└────────────────────────────────────────────────────────────────┴───────────────────┘

Files to create (new)

┌─────────────────────────────────────────────┬─────────────────────────────────────┐
│                    File                     │              Based on               │
├─────────────────────────────────────────────┼─────────────────────────────────────┤
│ source/.../config/unitree_g1/__init__.py    │ config/anymal_d/__init__.py         │
├─────────────────────────────────────────────┼─────────────────────────────────────┤
│ source/.../config/unitree_g1/flat_env_cfg.p │ config/anymal_d/flat_env_cfg.py     │
│ y                                           │                                     │
├─────────────────────────────────────────────┼─────────────────────────────────────┤
│ source/.../config/unitree_g1/agents/__init_ │ empty                               │
│ _.py                                        │                                     │
├─────────────────────────────────────────────┼─────────────────────────────────────┤
│ source/.../config/unitree_g1/agents/rsl_rl_ │ config/anymal_d/agents/rsl_rl_ppo_c │
│ ppo_cfg.py                                  │ fg.py                               │
├─────────────────────────────────────────────┼─────────────────────────────────────┤
│ source/.../config/unitree_g1/envs/unitree_g │ config/anymal_d/envs/anymal_d_manag │
│ 1_manager_based_mbrl_env.py                 │ er_based_mbrl_env.py                │
├─────────────────────────────────────────────┼─────────────────────────────────────┤
│ source/.../config/unitree_g1/envs/unitree_g │ config/anymal_d/envs/anymal_d_manag │
│ 1_manager_based_visualize_env.py            │ er_based_visualize_env.py           │
├─────────────────────────────────────────────┼─────────────────────────────────────┤
│ scripts/.../envs/unitree_g1_flat.py         │ scripts/.../envs/anymal_d_flat.py   │
├─────────────────────────────────────────────┼─────────────────────────────────────┤
│ scripts/.../configs/unitree_g1_flat_cfg.py  │ scripts/.../configs/anymal_d_flat_c │
│                                             │ fg.py                               │
└─────────────────────────────────────────────┴─────────────────────────────────────┘

---
Verification

1. Sanity check: Run Template-Isaac-Velocity-Flat-G1-Init-v0 in Isaac Sim to verify the
G1 spawns and walks
2. Data collection: Train a model-free policy with the Init env, collect trajectories
3. Normalizer tuning: Compute empirical mean/std from collected data, update configs
4. Pretrain: Train the world model with Template-Isaac-Velocity-Flat-G1-Pretrain-v0
5. Finetune: Train policy with MBPO using Template-Isaac-Velocity-Flat-G1-Finetune-v0
6. Visualize: Compare real vs imagined rollouts with the Visualize env

No changes needed to the base training scripts — they are robot-agnostic and use env
name to look up configs.

1. unitree_g1/__init__.py + agents/__init__.py (gym registrations)
2. flat_env_cfg.py (environment config)
3. unitree_g1_manager_based_mbrl_env.py (imagination env)
4. unitree_g1_manager_based_visualize_env.py (visualization env)
5. agents/rsl_rl_ppo_cfg.py (training configs)
6. unitree_g1_flat.py (offline training env)
7. unitree_g1_flat_cfg.py (offline training config)
8. Modify existing __init__.py files (imports)