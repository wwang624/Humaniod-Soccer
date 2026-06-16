from soccer.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import (
    G1FlatLowFreqPPORunnerCfg,
    G1FlatPPORunnerCfg,
    G1FlatRecurrentPPORunnerCfg,
    G1FlatStudentTeacherPPORunnerCfg,
    LOW_FREQ_SCALE,
)


def _apply_saya_runner_overrides(cfg) -> None:
    cfg.experiment_name = "saya_29dof_flat"
    cfg.max_iterations = 30000
    cfg.save_interval = 1000


class Saya29DoFFlatPPORunnerCfg(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super_post = getattr(super(), "__post_init__", None)
        if super_post is not None:
            super_post()
        _apply_saya_runner_overrides(self)


class Saya29DoFFlatRecurrentPPORunnerCfg(G1FlatRecurrentPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_saya_runner_overrides(self)


class Saya29DoFFlatLowFreqPPORunnerCfg(G1FlatLowFreqPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        _apply_saya_runner_overrides(self)


class Saya29DoFStudentTeacherPPORunnerCfg(G1FlatStudentTeacherPPORunnerCfg):
    def __post_init__(self):
        super_post = getattr(super(), "__post_init__", None)
        if super_post is not None:
            super_post()
        _apply_saya_runner_overrides(self)
