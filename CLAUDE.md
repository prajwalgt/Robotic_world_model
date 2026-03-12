# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This repository contains two interconnected Python packages implementing **Robotic World Model (RWM)** — a framework for joint training of neural dynamics models and locomotion policies, enabling offline policy training without a simulator.

- **`rsl_rl_rwm/`** — PyTorch RL library (fork of RSL RL) with model-based extensions
- **`robotic_world_model/`** — Isaac Lab extension for ANYmal D quadruped locomotion tasks

## Installation

```bash
# Install the RL library
cd rsl_rl_rwm && pip install -e .

# Install the Isaac Lab extension (requires Isaac Sim 4.5.0 + Isaac Lab 2.1.0)
python -m pip install -e robotic_world_model/source/mbrl
```

## Development Commands

```bash
# Run pre-commit checks (formatting, linting)
pre-commit run --all-files

# Individual formatters
black --line-length 120 --preview <file>
isort --profile black <file>
flake8 <file>
```

## Training Scripts

```bash
# Online RWM training (requires Isaac Sim)
python robotic_world_model/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=<task_name> --num_envs=<N>

# Offline policy training (no simulator needed, uses pretrained dynamics)
python robotic_world_model/scripts/reinforcement_learning/model_based/train.py

# Evaluate trained policy
python robotic_world_model/scripts/reinforcement_learning/rsl_rl/play.py

# Visualize imagination rollouts
python robotic_world_model/scripts/reinforcement_learning/rsl_rl/visualize.py
```

## Architecture

### Training Phases

1. **Pretrain (online):** Collect real environment transitions → train dynamics model ensemble → update policy with real data
2. **Finetune (imagination):** Use learned dynamics to generate imagined rollouts → train policy via MBPO-PPO → optionally collect more real data
3. **Offline:** Skip the simulator entirely — train policy purely on imagined rollouts from a static dynamics model (RWM-U)

### Key Components in `rsl_rl_rwm/`

| Module | Purpose |
|---|---|
| `rsl_rl/modules/system_dynamics.py` | Neural dynamics ensemble with GRU; produces state predictions + aleatoric uncertainty estimates |
| `rsl_rl/algorithms/mbpo_ppo.py` | Model-Based PPO algorithm that mixes real and imagined rollouts |
| `rsl_rl/runners/mbpo_on_policy_runner.py` | Full training loop managing pretrain/finetune phases and replay buffer |
| `rsl_rl/runners/on_policy_runner.py` | Standard PPO training loop (baseline) |
| `rsl_rl/storage/replay_buffer.py` | Stores real transitions for dynamics model training |
| `rsl_rl/modules/actor_critic.py` | Policy/value network |
| `rsl_rl/modules/rnd.py` | Random Network Distillation for exploration bonuses |

### Key Components in `robotic_world_model/`

| Module | Purpose |
|---|---|
| `mbrl/envs/manager_based_mbrl_env.py` | **Imagination environment** — wraps the dynamics model to produce policy-compatible observations and rewards without a physics simulator |
| `mbrl/envs/manager_based_visualize_env.py` | Renders imagination rollouts for debugging |
| `mbrl/tasks/.../anymal_d/flat_env_cfg.py` | Full Isaac Lab environment config for ANYmal D flat terrain |
| `mbrl/tasks/.../anymal_d/agents/rsl_rl_ppo_cfg.py` | Training hyperparameters: dynamics ensemble config, MBPO runner config, imagination settings |
| `mbrl/rl/rsl_rl/rl_cfg.py` | Dataclasses for all training configuration |

### Dynamics Model (`SystemDynamics`)

- **Input:** history of observations (horizon=32 steps) + action
- **Architecture:** GRU encoder with configurable hidden size (default 256 for ANYmal D 45-dim state, 12-dim action)
- **Output:** next state prediction + optional privileged outputs (contacts, terminations, rewards)
- **Ensemble:** multiple heads sharing GRU backbone; ensemble disagreement measures epistemic uncertainty
- **Uncertainty penalty:** configurable weights applied during imagination rollouts to penalize high-uncertainty regions

### Configuration

Training configs are Python dataclasses in `rsl_rl/utils/utils.py` and `mbrl/rl/rsl_rl/rl_cfg.py`. The primary config file for ANYmal D is:
`robotic_world_model/source/mbrl/mbrl/tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py`

Key hyperparameters:
- `ensemble_size`: number of dynamics model heads
- `history_horizon`: steps of observation history fed to GRU (default 32)
- `imagination_horizon`: rollout length during MBPO imagination phase
- `uncertainty_penalty_*`: weights on aleatoric/epistemic uncertainty terms

### Logging

Supports TensorBoard, Weights & Biases, and Neptune. Configure via runner cfg.

## Code Style

- Line length: **120 characters**
- Formatter: **Black** (preview mode)
- Import sorting: **isort** with black profile
- Python: 3.10+ (mbrl extension), 3.8+ (rsl_rl_rwm)
