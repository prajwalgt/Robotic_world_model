import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Template-Isaac-Velocity-Flat-G1-Init-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_INIT",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Velocity-Flat-G1-Pretrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_PRETRAIN",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOPretrainRunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Velocity-Flat-G1-Finetune-v0",
    entry_point="mbrl.tasks.manager_based.locomotion.velocity.config.unitree_g1.envs.unitree_g1_manager_based_mbrl_env:UnitreeG1ManagerBasedMBRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_FINETUNE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOFinetuneRunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Velocity-Flat-G1-Visualize-v0",
    entry_point="mbrl.tasks.manager_based.locomotion.velocity.config.unitree_g1.envs.unitree_g1_manager_based_visualize_env:UnitreeG1ManagerBasedVisualizeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg_VISUALIZE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPOVisualizeRunnerCfg",
    },
)
