from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import math

from soccer.robots.saya_29dof import (
    SAYA_29DOF_ANCHOR_BODY_NAME,
    SAYA_29DOF_FEET_BODY_NAMES,
    SAYA_29DOF_ROOT_BODY_NAME,
    SAYA_29DOF_TRACKING_BODY_NAMES,
)
from soccer.tasks.tracking import mdp
from soccer.tasks.tracking.config.g1.soccer_flat_env_cfg import (
    G1FlatKickEnvCfg,
    G1FlatKickMovingEnvCfg,
    G1FlatMotionEnvCfg,
    G1FlatSoccerBlindEnvCfg,
    G1FlatSoccerSceneCfg,
    G1FlatSoccerStudentEnvCfg,
    G1TerrainMotionEnvCfg,
)
from soccer.tasks.tracking.config.saya_29dof.flat_env_cfg import apply_saya_29dof_base_overrides


def _apply_saya_soccer_overrides(cfg) -> None:
    cfg.scene.robot = cfg.scene.robot.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.commands.motion.anchor_body_name = SAYA_29DOF_ANCHOR_BODY_NAME
    cfg.commands.motion.soccer_obs_body_name = SAYA_29DOF_ANCHOR_BODY_NAME
    cfg.commands.motion.body_names = list(SAYA_29DOF_TRACKING_BODY_NAMES)
    cfg.commands.motion.destination_mode = "ball_anchor"
    cfg.commands.motion.ball_to_goal_vector = (2.117595, 0.0, 0.0)
    cfg.commands.motion.destination_center = (2.117595, 0.0, 0.11)
    cfg.commands.motion.destination_length = 0.0
    cfg.commands.motion.destination_width = 0.0
    cfg.commands.motion.ball_placement_mode = "kick_foot"
    cfg.commands.motion.ball_contact_phase = 0.9
    cfg.commands.motion.valid_contact_phase_range = (0.6, 1.0)
    cfg.commands.motion.pose_range["yaw"] = (-math.pi, math.pi)
    cfg.commands.motion.rotate_soccer_targets_with_root_yaw = True
    cfg.commands.motion.curve_offset_range = {
        "radius": (-0.05, 0.05),
        "arc_angle": 0.05,
        "height": 0.11,
    }
    # if hasattr(cfg.rewards, "motion_global_anchor_ori"):
    #     cfg.rewards.motion_global_anchor_ori.weight = 0.0

    cfg.foot_cfg = SceneEntityCfg("robot", body_names=list(SAYA_29DOF_FEET_BODY_NAMES))
    cfg.waist_cfg = SceneEntityCfg(
        "robot",
        joint_names=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
    )

    if hasattr(cfg.rewards, "motion_body_pos"):
        cfg.rewards.motion_body_pos.params["body_names"] = [
            SAYA_29DOF_ROOT_BODY_NAME,
            "left_hip_roll_link",
            "left_knee_link",
            "right_hip_roll_link",
            "right_knee_link",
            "torso_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "left_wrist_pitch_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
            "right_wrist_pitch_link",
        ]
    if hasattr(cfg, "motion_body_ori"):
        cfg.motion_body_ori.params["body_names"] = cfg.rewards.motion_body_pos.params["body_names"]
    if hasattr(cfg.rewards, "motion_foot_pos"):
        cfg.rewards.motion_foot_pos.params["foot_body_names"] = list(SAYA_29DOF_FEET_BODY_NAMES)
    if hasattr(cfg.rewards, "foot_distance"):
        cfg.rewards.foot_distance.params["foot_cfg"] = cfg.foot_cfg
    if hasattr(cfg.rewards, "waist_action_rate_l2"):
        cfg.rewards.waist_action_rate_l2.params["waist_cfg"] = cfg.waist_cfg
    for name in ("target_point_contact", "sideways_kick", "ball_velocity_direction_alignment", "ball_speed_reward"):
        term = getattr(cfg.rewards, name, None)
        if term is not None and "foot_cfg" in term.params:
            term.params["foot_cfg"] = cfg.foot_cfg


@configclass
class Saya29DoFFlatSoccerSceneCfg(G1FlatSoccerSceneCfg):
    pass


@configclass
class Saya29DoFTerrainMotionEnvCfg(G1TerrainMotionEnvCfg):
    scene: Saya29DoFFlatSoccerSceneCfg = Saya29DoFFlatSoccerSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        apply_saya_29dof_base_overrides(self)
        _apply_saya_soccer_overrides(self)


@configclass
class Saya29DoFFlatMotionEnvCfg(G1FlatMotionEnvCfg):
    scene: Saya29DoFFlatSoccerSceneCfg = Saya29DoFFlatSoccerSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        apply_saya_29dof_base_overrides(self)
        _apply_saya_soccer_overrides(self)


@configclass
class Saya29DoFFlatKickEnvCfg(G1FlatKickEnvCfg):
    scene: Saya29DoFFlatSoccerSceneCfg = Saya29DoFFlatSoccerSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        apply_saya_29dof_base_overrides(self)
        _apply_saya_soccer_overrides(self)


@configclass
class Saya29DoFFlatKickMovingEnvCfg(G1FlatKickMovingEnvCfg):
    scene: Saya29DoFFlatSoccerSceneCfg = Saya29DoFFlatSoccerSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        apply_saya_29dof_base_overrides(self)
        _apply_saya_soccer_overrides(self)


@configclass
class Saya29DoFFlatSoccerBlindEnvCfg(G1FlatSoccerBlindEnvCfg):
    scene: Saya29DoFFlatSoccerSceneCfg = Saya29DoFFlatSoccerSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        apply_saya_29dof_base_overrides(self)
        _apply_saya_soccer_overrides(self)


@configclass
class Saya29DoFFlatSoccerStudentEnvCfg(Saya29DoFFlatKickEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        student_obs = self.observations.policy.copy()
        teacher_obs = self.observations.policy.copy()
        teacher_obs.enable_corruption = False
        student_obs.target_point_pos = ObsTerm(
            func=mdp.constant_target_point_pos,
            params={"command_name": "motion"},
        )
        student_obs.target_destination_pos_local = ObsTerm(
            func=mdp.target_destination_pos_local,
            params={"command_name": "motion"},
        )
        self.observations.teacher = teacher_obs
        self.observations.policy = student_obs


@configclass
class Saya29DoFFlatSoccerDistillEnvCfg(Saya29DoFFlatSoccerStudentEnvCfg):
    """Dedicated Saya soccer distillation environment with looser rollout terminations."""

    def __post_init__(self):
        super().__post_init__()
        soccer_target_noise_params = {
            "command_name": "motion",
            "ball_noise_std": (0.03, 0.03, 0.02),
            "goal_noise_std": (0.03, 0.03, 0.02),
            "noise_type": "normal",
            "update_interval": 2,
            "dropout_prob": 0.10,
            "hold_last": True,
        }
        self.observations.policy.target_point_pos = ObsTerm(
            func=mdp.noisy_target_point_pos,
            params=soccer_target_noise_params,
        )
        self.observations.policy.target_destination_pos_local = ObsTerm(
            func=mdp.noisy_target_destination_pos_local,
            params=soccer_target_noise_params,
        )
        self.terminations.anchor_pos_z.params["threshold"] = 0.15
        self.terminations.anchor_ori = None
        self.terminations.ee_body_pos = None
