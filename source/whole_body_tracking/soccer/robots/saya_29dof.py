from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from soccer.assets import ASSET_DIR

SAYA_29DOF_SOURCE_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
)

SAYA_29DOF_TRACKING_BODY_NAMES: tuple[str, ...] = (
    "base_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_pitch_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_pitch_link",
)

SAYA_29DOF_FEET_BODY_NAMES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")
SAYA_29DOF_ROOT_BODY_NAME = "base_link"
SAYA_29DOF_ANCHOR_BODY_NAME = "torso_link"

_PROJECT_SAYA_URDF = (
    Path(ASSET_DIR)
    / "saya_description"
    / "urdf"
    / "v1.1_u3.0_0303_v0_29dof_no_duplicate_mesh_collision.urdf"
)
SAYA_29DOF_URDF = Path(os.environ.get("SOCCER_SAYA_29DOF_URDF", str(_PROJECT_SAYA_URDF)))

_PROJECT_SAYA_MJCF = (
    Path(ASSET_DIR) / "saya_description" / "urdf" / "v1.1_u3.0_0303_v0_29dof_floating.xml"
)
SAYA_29DOF_MJCF = Path(os.environ.get("SOCCER_SAYA_29DOF_MJCF", str(_PROJECT_SAYA_MJCF)))

ROTOR_INERTIA_EC_A10020_P1_12 = 485.5e-6
ROTOR_INERTIA_EC_A8112_P1_18 = 149.22e-6
ROTOR_INERTIA_EC_A6408_P2_25 = 62.4e-6
ROTOR_INERTIA_EC_A4310_P2_36 = 18.6e-6
ROTOR_INERTIA_17_SERIES = 0.072e-4
ROTOR_INERTIA_SHD11 = 0.016e-4

ARMATURE_EC_A10020_P1_12 = ROTOR_INERTIA_EC_A10020_P1_12 * 12**2
ARMATURE_EC_A8112_P1_18 = ROTOR_INERTIA_EC_A8112_P1_18 * 18**2
ARMATURE_EC_A6408_P2_25 = ROTOR_INERTIA_EC_A6408_P2_25 * 25**2
ARMATURE_EC_A4310_P2_36 = ROTOR_INERTIA_EC_A4310_P2_36 * 36**2
ARMATURE_17_SERIES_120 = ROTOR_INERTIA_17_SERIES * 120**2
ARMATURE_17_SERIES_100 = ROTOR_INERTIA_17_SERIES * 100**2
ARMATURE_SHD11_100 = ROTOR_INERTIA_SHD11 * 100**2

NATURAL_FREQ = 7 * 2.0 * 3.1415926535
DAMPING_RATIO = 1.3

STIFFNESS_SAYA29_LEG_MAIN = ARMATURE_EC_A10020_P1_12 * NATURAL_FREQ**2
STIFFNESS_SAYA29_YAW = ARMATURE_EC_A8112_P1_18 * NATURAL_FREQ**2
STIFFNESS_SAYA29_ANKLE = ARMATURE_EC_A4310_P2_36 * NATURAL_FREQ**2
STIFFNESS_SAYA29_WAIST_RP = ARMATURE_EC_A6408_P2_25 * NATURAL_FREQ**2
STIFFNESS_SAYA29_SHOULDER_PITCH_ROLL = ARMATURE_17_SERIES_120 * NATURAL_FREQ**2
STIFFNESS_SAYA29_SHOULDER_YAW_ELBOW = ARMATURE_17_SERIES_100 * NATURAL_FREQ**2
STIFFNESS_SAYA29_WRIST = ARMATURE_SHD11_100 * NATURAL_FREQ**2

DAMPING_SAYA29_LEG_MAIN = 2.0 * DAMPING_RATIO * ARMATURE_EC_A10020_P1_12 * NATURAL_FREQ
DAMPING_SAYA29_YAW = 2.0 * DAMPING_RATIO * ARMATURE_EC_A8112_P1_18 * NATURAL_FREQ
DAMPING_SAYA29_ANKLE = 2.0 * DAMPING_RATIO * ARMATURE_EC_A4310_P2_36 * NATURAL_FREQ
DAMPING_SAYA29_WAIST_RP = 2.0 * DAMPING_RATIO * ARMATURE_EC_A6408_P2_25 * NATURAL_FREQ
DAMPING_SAYA29_SHOULDER_PITCH_ROLL = 2.0 * DAMPING_RATIO * ARMATURE_17_SERIES_120 * NATURAL_FREQ
DAMPING_SAYA29_SHOULDER_YAW_ELBOW = 2.0 * DAMPING_RATIO * ARMATURE_17_SERIES_100 * NATURAL_FREQ
DAMPING_SAYA29_WRIST = 2.0 * DAMPING_RATIO * ARMATURE_SHD11_100 * NATURAL_FREQ

SAYA_29DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=str(SAYA_29DOF_URDF),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            ".*_hip_pitch_joint": -0.11,
            ".*_knee_joint": 0.25,
            ".*_ankle_pitch_joint": -0.14,
            "left_shoulder_roll_joint": 0.15,
            "right_shoulder_roll_joint": -0.15,
            ".*_shoulder_pitch_joint": 0.15,
            ".*_elbow_joint": 1.10,
            "left_wrist_roll_joint": 0.15,
            "right_wrist_roll_joint": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "legs_hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint"],
            effort_limit_sim=150.0,
            velocity_limit_sim=13.4041,
            stiffness=STIFFNESS_SAYA29_LEG_MAIN,
            damping=DAMPING_SAYA29_LEG_MAIN,
            armature=ARMATURE_EC_A10020_P1_12,
            friction=0.05,
        ),
        "legs_hip_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint"],
            effort_limit_sim=150.0,
            velocity_limit_sim=13.4041,
            stiffness=STIFFNESS_SAYA29_LEG_MAIN,
            damping=DAMPING_SAYA29_LEG_MAIN,
            armature=ARMATURE_EC_A10020_P1_12,
            friction=0.05,
        ),
        "legs_hip_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint"],
            effort_limit_sim=90.0,
            velocity_limit_sim=16.1268,
            stiffness=STIFFNESS_SAYA29_YAW,
            damping=DAMPING_SAYA29_YAW,
            armature=ARMATURE_EC_A8112_P1_18,
            friction=0.05,
        ),
        "legs_knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_knee_joint"],
            effort_limit_sim=150.0,
            velocity_limit_sim=13.4041,
            stiffness=STIFFNESS_SAYA29_LEG_MAIN,
            damping=DAMPING_SAYA29_LEG_MAIN,
            armature=ARMATURE_EC_A10020_P1_12,
            friction=0.05,
        ),
        "feet_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint"],
            effort_limit_sim=108.0,
            velocity_limit_sim=8.0634,
            stiffness=2.0 * STIFFNESS_SAYA29_ANKLE,
            damping=2.0 * DAMPING_SAYA29_ANKLE,
            armature=2.0 * ARMATURE_EC_A4310_P2_36,
            friction=0.05,
        ),
        "feet_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint"],
            effort_limit_sim=108.0,
            velocity_limit_sim=8.0634,
            stiffness=2.0 * STIFFNESS_SAYA29_ANKLE,
            damping=2.0 * DAMPING_SAYA29_ANKLE,
            armature=2.0 * ARMATURE_EC_A4310_P2_36,
            friction=0.05,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=90.0,
            velocity_limit_sim=16.1268,
            stiffness=STIFFNESS_SAYA29_YAW,
            damping=DAMPING_SAYA29_YAW,
            armature=ARMATURE_EC_A8112_P1_18,
            friction=0.05,
        ),
        "waist_roll": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint"],
            effort_limit_sim=180.0,
            velocity_limit_sim=13.1947,
            stiffness=2.0 * STIFFNESS_SAYA29_WAIST_RP,
            damping=2.0 * DAMPING_SAYA29_WAIST_RP,
            armature=2.0 * ARMATURE_EC_A6408_P2_25,
            friction=0.05,
        ),
        "waist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["waist_pitch_joint"],
            effort_limit_sim=180.0,
            velocity_limit_sim=13.1947,
            stiffness=2.0 * STIFFNESS_SAYA29_WAIST_RP,
            damping=2.0 * DAMPING_SAYA29_WAIST_RP,
            armature=2.0 * ARMATURE_EC_A6408_P2_25,
            friction=0.05,
        ),
        "shoulder_pitch_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint"],
            effort_limit_sim=80.0,
            velocity_limit_sim=2.618,
            stiffness=STIFFNESS_SAYA29_SHOULDER_PITCH_ROLL,
            damping=DAMPING_SAYA29_SHOULDER_PITCH_ROLL,
            armature=ARMATURE_17_SERIES_120,
            friction=0.05,
        ),
        "shoulder_yaw_elbow": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_yaw_joint", ".*_elbow_joint"],
            effort_limit_sim=80.0,
            velocity_limit_sim=3.1416,
            stiffness=STIFFNESS_SAYA29_SHOULDER_YAW_ELBOW,
            damping=DAMPING_SAYA29_SHOULDER_YAW_ELBOW,
            armature=ARMATURE_17_SERIES_100,
            friction=0.05,
        ),
        "wrist_pitch_roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_roll_joint"],
            effort_limit_sim=11.5,
            velocity_limit_sim=3.1416,
            stiffness=STIFFNESS_SAYA29_WRIST,
            damping=DAMPING_SAYA29_WRIST,
            armature=ARMATURE_SHD11_100,
            friction=0.05,
        ),
        "wrist_yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_yaw_joint"],
            effort_limit_sim=11.5,
            velocity_limit_sim=3.1416,
            stiffness=STIFFNESS_SAYA29_WRIST,
            damping=DAMPING_SAYA29_WRIST,
            armature=ARMATURE_SHD11_100,
            friction=0.05,
        ),
    },
)

SAYA_29DOF_ACTION_SCALE = {}
for actuator in SAYA_29DOF_CFG.actuators.values():
    effort = actuator.effort_limit_sim
    stiffness = actuator.stiffness
    names = actuator.joint_names_expr
    if not isinstance(effort, dict):
        effort = {name: effort for name in names}
    if not isinstance(stiffness, dict):
        stiffness = {name: stiffness for name in names}
    for name in names:
        if name in effort and name in stiffness and stiffness[name]:
            SAYA_29DOF_ACTION_SCALE[name] = 0.25 * effort[name] / stiffness[name]
