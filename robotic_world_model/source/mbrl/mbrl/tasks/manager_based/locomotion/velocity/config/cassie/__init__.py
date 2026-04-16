import gymnasium as gym

from . import agents


gym.register(
    id="Template-Isaac-Velocity-Flat-Cassie-Init-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:CassieFlatEnvCfg_INIT",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieFlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Velocity-Flat-Cassie-Pretrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:CassieFlatEnvCfg_PRETRAIN",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieFlatPPOPretrainRunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Velocity-Flat-Cassie-Finetune-v0",
    entry_point="mbrl.tasks.manager_based.locomotion.velocity.config.cassie.envs.cassie_manager_based_mbrl_env:CassieManagerBasedMBRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:CassieFlatEnvCfg_FINETUNE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieFlatPPOFinetuneRunnerCfg",
    },
)

gym.register(
    id="Template-Isaac-Velocity-Flat-Cassie-Visualize-v0",
    entry_point="mbrl.tasks.manager_based.locomotion.velocity.config.cassie.envs.cassie_manager_based_visualize_env:CassieManagerBasedVisualizeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:CassieFlatEnvCfg_VISUALIZE",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CassieFlatPPOVisualizeRunnerCfg",
    },
)
