from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from soccer.robots.saya_29dof import (
    SAYA_29DOF_ACTION_SCALE,
    SAYA_29DOF_ANCHOR_BODY_NAME,
    SAYA_29DOF_CFG,
    SAYA_29DOF_FEET_BODY_NAMES,
    SAYA_29DOF_ROOT_BODY_NAME,
    SAYA_29DOF_TRACKING_BODY_NAMES,
)
from soccer.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from soccer.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg


def apply_saya_29dof_base_overrides(cfg) -> None:
    cfg.scene.robot = SAYA_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.actions.joint_pos.scale = SAYA_29DOF_ACTION_SCALE
    cfg.commands.motion.anchor_body_name = SAYA_29DOF_ANCHOR_BODY_NAME
    cfg.commands.motion.soccer_obs_body_name = SAYA_29DOF_ANCHOR_BODY_NAME
    cfg.commands.motion.body_names = list(SAYA_29DOF_TRACKING_BODY_NAMES)
    cfg.commands.motion.debug_vis = False
    cfg.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
        r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)(?!left_hand_link$)(?!right_hand_link$).+$"
    ]

    cfg.events.base_com.params["asset_cfg"].body_names = [SAYA_29DOF_ANCHOR_BODY_NAME]
    cfg.terminations.ee_body_pos.params["body_names"] = [
        *SAYA_29DOF_FEET_BODY_NAMES,
        "left_wrist_pitch_link",
        "right_wrist_pitch_link",
    ]
    cfg.viewer.asset_name = "robot"
    cfg.viewer.body_name = SAYA_29DOF_ANCHOR_BODY_NAME


@configclass
class Saya29DoFFlatEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        apply_saya_29dof_base_overrides(self)


@configclass
class Saya29DoFFlatWoStateEstimationEnvCfg(Saya29DoFFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class Saya29DoFFlatLowFreqEnvCfg(Saya29DoFFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE


SAYA_ROOT_BODY_CFG = SceneEntityCfg("robot", body_names=[SAYA_29DOF_ROOT_BODY_NAME])
