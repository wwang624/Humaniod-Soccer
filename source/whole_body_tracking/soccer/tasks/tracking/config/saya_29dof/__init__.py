import gymnasium as gym

from . import agents, flat_env_cfg, soccer_flat_env_cfg


gym.register(
    id="Tracking-Flat-Saya29DoF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Saya29DoFFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Saya29DoFFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Saya29DoF-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Saya29DoFFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Saya29DoFFlatRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Saya29DoF-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Saya29DoFFlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Saya29DoFFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Saya29DoF-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Saya29DoFFlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Saya29DoFFlatLowFreqPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Terrain-Saya29DoF-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.Saya29DoFTerrainMotionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Saya29DoFFlatRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Saya29DoF-Motion-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.Saya29DoFFlatMotionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Saya29DoFFlatRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Saya29DoF-SoccerDestination-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.Saya29DoFFlatKickEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Saya29DoFFlatRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Saya29DoF-SoccerMoving-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.Saya29DoFFlatKickMovingEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Saya29DoFFlatRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Saya29DoF-SoccerBlind-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.Saya29DoFFlatSoccerBlindEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Saya29DoFFlatRecurrentPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Saya29DoF-Soccer-Distillation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": soccer_flat_env_cfg.Saya29DoFFlatSoccerDistillEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Saya29DoFStudentTeacherPPORunnerCfg",
    },
)
