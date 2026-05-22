#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

SPACE_KEY = 32
ENTER_KEY = 257
R_KEY_LOWER = 114
R_KEY_UPPER = 82
W_KEY_LOWER = 119
W_KEY_UPPER = 87
S_KEY_LOWER = 115
S_KEY_UPPER = 83
A_KEY_LOWER = 97
A_KEY_UPPER = 65
D_KEY_LOWER = 100
D_KEY_UPPER = 68
Q_KEY_LOWER = 113
Q_KEY_UPPER = 81
E_KEY_LOWER = 101
E_KEY_UPPER = 69


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a soccer ONNX policy in MuJoCo sim2sim.")
    parser.add_argument("--onnx-path", required=True, help="Exported ONNX policy path.")
    parser.add_argument(
        "--xml-path",
        default="source/whole_body_tracking/soccer/assets/unitree_description/mjcf/g1_actuator.xml",
        help="MuJoCo XML path.",
    )
    parser.add_argument("--sim-dt", type=float, default=0.005, help="MuJoCo physics dt.")
    parser.add_argument("--control-decimation", type=int, default=4, help="Physics steps per policy step.")
    parser.add_argument("--simulation-duration", type=float, default=600.0, help="Max wall-clock runtime in viewer.")
    parser.add_argument("--ball-body-name", type=str, default="largebox", help="MuJoCo ball body/joint name.")
    parser.add_argument("--kick-speed-threshold", type=float, default=0.8, help="Planar ball-speed threshold for reset.")
    parser.add_argument("--kick-reset-delay-steps", type=int, default=250, help="Policy steps to wait after kick before reset.")
    parser.add_argument("--kick-armed-min-steps", type=int, default=20, help="Do not arm kick-reset logic before this many policy steps.")
    parser.add_argument("--fall-height-threshold", type=float, default=0.45, help="Reset if pelvis height drops below this.")
    parser.add_argument("--max-episode-steps", type=int, default=500, help="Optional hard episode-step limit. Matches the training env by default.")
    parser.add_argument("--ball-height", type=float, default=0.11, help="Ball spawn height.")
    parser.add_argument("--ball-arc-angle", type=float, default=math.pi / 9.0, help="Ball spawn arc half-angle.")
    parser.add_argument("--ball-radius-offset-min", type=float, default=-0.25, help="Min radius offset for static ball spawn.")
    parser.add_argument("--ball-radius-offset-max", type=float, default=0.25, help="Max radius offset for static ball spawn.")
    parser.add_argument("--goal-center-x", type=float, default=0.0, help="Goal center x in world coordinates.")
    parser.add_argument("--goal-center-y", type=float, default=-5.0, help="Goal center y in world coordinates.")
    parser.add_argument("--goal-center-z", type=float, default=0.11, help="Goal center z in world coordinates.")
    parser.add_argument("--goal-length", type=float, default=1.0, help="Goal sampling rectangle length.")
    parser.add_argument("--goal-width", type=float, default=0.5, help="Goal sampling rectangle width.")
    parser.add_argument("--viewer-distance", type=float, default=3.0, help="Viewer camera distance.")
    parser.add_argument("--viewer-elevation", type=float, default=-20.0, help="Viewer camera elevation.")
    parser.add_argument("--ball-radius", type=float, default=0.11, help="Radius of the injected MuJoCo ball.")
    parser.add_argument("--goal-marker-radius", type=float, default=0.11, help="Radius of the red goal marker sphere.")
    parser.add_argument("--manual-ball-step", type=float, default=0.05, help="Manual ball translation step size in meters.")
    parser.add_argument("--enable-soccer-noise", action="store_true", default=False, help="Inject observation noise into ball/goal local coordinates.")
    parser.add_argument("--ball-noise-base-std", type=float, default=0.01, help="Base Gaussian std for ball local-position noise in meters.")
    parser.add_argument("--ball-noise-dist-coeff", type=float, default=0.02, help="Additional std per meter of ball distance.")
    parser.add_argument("--ball-noise-vel-coeff", type=float, default=0.05, help="Additional std per m/s of ball planar speed.")
    parser.add_argument("--goal-noise-base-std", type=float, default=0.005, help="Base Gaussian std for goal local-position noise in meters.")
    parser.add_argument("--goal-noise-dist-coeff", type=float, default=0.01, help="Additional std per meter of goal distance.")
    parser.add_argument("--soccer-noise-print-interval", type=int, default=25, help="Print current soccer-noise strength every N policy steps when enabled.")
    return parser.parse_args()


def csv_to_list(raw: str) -> list[str]:
    if raw is None or raw == "":
        return []
    return [item for item in raw.split(",") if item != ""]


def decode_metadata_list(raw: str) -> list[str]:
    if raw is None or raw == "":
        return []
    raw = raw.strip()
    if raw.startswith("["):
        return list(json.loads(raw))
    return csv_to_list(raw)


def decode_metadata_array(raw: str) -> np.ndarray:
    if raw is None or raw == "":
        return np.zeros((0,), dtype=np.float32)
    raw = raw.strip()
    if raw.startswith("["):
        return np.asarray(json.loads(raw), dtype=np.float32)
    return csv_to_float_array(raw)


def csv_to_float_array(raw: str) -> np.ndarray:
    values = csv_to_list(raw)
    if not values:
        return np.array([], dtype=np.float32)
    return np.asarray([float(v) for v in values], dtype=np.float32)


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_inv_wxyz(q: np.ndarray) -> np.ndarray:
    norm = np.dot(q, q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32) / norm


def quat_rotate_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qvec = q[1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2.0 * (q[0] * uv + uuv)


def quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return quat_rotate_wxyz(quat_inv_wxyz(q), v)


@dataclass
class ModelMetadata:
    joint_names: list[str]
    default_joint_pos: np.ndarray
    joint_stiffness: np.ndarray
    joint_damping: np.ndarray
    action_scale: np.ndarray
    observation_names: list[str]
    anchor_body_name: str
    body_names: list[str]
    motion_names: list[str]
    motion_lengths: np.ndarray
    motion_kick_leg_names: list[str]
    final_anchor_pos: np.ndarray


class OnnxSoccerPolicy:
    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.output_index = {name: idx for idx, name in enumerate(self.output_names)}
        self.model_meta = self.session.get_modelmeta().custom_metadata_map
        self.uses_motion_index = "motion_idx" in self.input_names
        self._validate_export_contract(onnx_path)
        self.metadata = ModelMetadata(
            joint_names=decode_metadata_list(self.model_meta.get("joint_names", "")),
            default_joint_pos=csv_to_float_array(self.model_meta.get("default_joint_pos", "")),
            joint_stiffness=csv_to_float_array(self.model_meta.get("joint_stiffness", "")),
            joint_damping=csv_to_float_array(self.model_meta.get("joint_damping", "")),
            action_scale=csv_to_float_array(self.model_meta.get("action_scale", "")),
            observation_names=decode_metadata_list(self.model_meta.get("observation_names", "")),
            anchor_body_name=self.model_meta.get("anchor_body_name", "torso_link"),
            body_names=decode_metadata_list(self.model_meta.get("body_names", "")),
            motion_names=decode_metadata_list(self.model_meta.get("motion_names", "")),
            motion_lengths=csv_to_float_array(self.model_meta.get("motion_lengths", "")),
            motion_kick_leg_names=decode_metadata_list(self.model_meta.get("motion_kick_leg_names", "")),
            final_anchor_pos=decode_metadata_array(self.model_meta.get("final_anchor_pos", "")),
        )
        self.obs_dim = int(self.session.get_inputs()[0].shape[-1])
        self.is_recurrent = {"h_in", "c_in", "time_step"}.issubset(self.input_names)
        self.recurrent_shape = self._infer_recurrent_shape()
        self.reference = self._precompute_reference_cache()

    def _validate_export_contract(self, onnx_path: str):
        required_meta = {
            "joint_names",
            "default_joint_pos",
            "joint_stiffness",
            "joint_damping",
            "action_scale",
            "observation_names",
            "anchor_body_name",
            "body_names",
        }
        missing_meta = sorted(key for key in required_meta if key not in self.model_meta)
        if missing_meta:
            raise RuntimeError(
                "ONNX model is missing HumanoidSoccer export metadata "
                f"{missing_meta}. Re-export the model with "
                "`soccer.utils.exporter.export_motion_policy_as_onnx()`. "
                f"Model path: {onnx_path}"
            )

        required_outputs = {
            "actions",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        }
        if self.uses_motion_index:
            required_outputs.update({"motion_length", "motion_idx_selected", "time_step_total"})
        else:
            required_outputs.add("time_step_total")
        missing_outputs = sorted(name for name in required_outputs if name not in self.output_index)
        if missing_outputs:
            raise RuntimeError(
                "ONNX model is missing required motion-reference outputs "
                f"{missing_outputs}. Use the motion exporter for soccer sim2sim. "
                f"Model path: {onnx_path}"
            )

        if "time_step" not in self.input_names:
            raise RuntimeError(
                "ONNX model is missing the `time_step` input required by the motion-conditioned "
                f"runtime bridge. Model path: {onnx_path}"
            )

    def _infer_recurrent_shape(self) -> tuple[int, int] | None:
        if not self.is_recurrent:
            return None
        h_shape = self.session.get_inputs()[1].shape
        num_layers = int(h_shape[0])
        hidden_dim = int(h_shape[2])
        return num_layers, hidden_dim

    def _zero_inputs(self, time_step: int, motion_idx: int = 0) -> dict[str, np.ndarray]:
        inputs: dict[str, np.ndarray] = {"obs": np.zeros((1, self.obs_dim), dtype=np.float32)}
        if self.is_recurrent:
            assert self.recurrent_shape is not None
            num_layers, hidden_dim = self.recurrent_shape
            inputs["h_in"] = np.zeros((num_layers, 1, hidden_dim), dtype=np.float32)
            inputs["c_in"] = np.zeros((num_layers, 1, hidden_dim), dtype=np.float32)
        if self.uses_motion_index:
            inputs["motion_idx"] = np.array([[motion_idx]], dtype=np.float32)
        inputs["time_step"] = np.array([[time_step]], dtype=np.float32)
        return inputs

    def _run_raw(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        feed = {name: inputs[name] for name in self.input_names}
        return self.session.run(None, feed)

    def _precompute_reference_cache(self) -> dict[str, np.ndarray]:
        if self.uses_motion_index:
            if self.metadata.motion_lengths.size == 0 or not self.metadata.motion_names:
                raise RuntimeError("Bundled multi-motion ONNX is missing motion_names/motion_lengths metadata.")
            motion_count = len(self.metadata.motion_names)
            motion_lengths = self.metadata.motion_lengths.astype(np.int32)
            max_time = int(np.max(motion_lengths))
            cache: dict[str, np.ndarray] = {
                "joint_pos": np.zeros((motion_count, max_time, len(self.metadata.joint_names)), dtype=np.float32),
                "joint_vel": np.zeros((motion_count, max_time, len(self.metadata.joint_names)), dtype=np.float32),
                "body_pos_w": np.zeros((motion_count, max_time, len(self.metadata.body_names), 3), dtype=np.float32),
                "body_quat_w": np.zeros((motion_count, max_time, len(self.metadata.body_names), 4), dtype=np.float32),
                "body_lin_vel_w": np.zeros((motion_count, max_time, len(self.metadata.body_names), 3), dtype=np.float32),
                "body_ang_vel_w": np.zeros((motion_count, max_time, len(self.metadata.body_names), 3), dtype=np.float32),
            }
            for motion_idx in range(motion_count):
                for step in range(int(motion_lengths[motion_idx])):
                    outputs = self._run_raw(self._zero_inputs(step, motion_idx))
                    for key in cache:
                        cache[key][motion_idx, step] = outputs[self.output_index[key]].squeeze(0).astype(np.float32)
            return {
                "time_step_total": np.array(max_time, dtype=np.int32),
                "motion_lengths": motion_lengths,
                **cache,
            }

        first = self._run_raw(self._zero_inputs(0))
        time_step_total = int(first[self.output_index["time_step_total"]].reshape(-1)[0])
        cache_single: dict[str, list[np.ndarray]] = {
            "joint_pos": [],
            "joint_vel": [],
            "body_pos_w": [],
            "body_quat_w": [],
            "body_lin_vel_w": [],
            "body_ang_vel_w": [],
        }
        for step in range(time_step_total):
            outputs = self._run_raw(self._zero_inputs(step))
            for key in cache_single:
                cache_single[key].append(outputs[self.output_index[key]].squeeze(0).astype(np.float32))

        return {
            "time_step_total": np.array(time_step_total, dtype=np.int32),
            **{key: np.stack(values, axis=0) for key, values in cache_single.items()},
        }

    def act(
        self,
        obs: np.ndarray,
        time_step: int,
        motion_idx: int = 0,
        h_in: np.ndarray | None = None,
        c_in: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        inputs: dict[str, np.ndarray] = {"obs": obs.astype(np.float32), "time_step": np.array([[time_step]], dtype=np.float32)}
        if self.uses_motion_index:
            inputs["motion_idx"] = np.array([[motion_idx]], dtype=np.float32)
        if self.is_recurrent:
            assert h_in is not None and c_in is not None
            inputs["h_in"] = h_in.astype(np.float32)
            inputs["c_in"] = c_in.astype(np.float32)
        outputs = self._run_raw(inputs)
        actions = outputs[self.output_index["actions"]].astype(np.float32)
        if self.is_recurrent:
            h_out = outputs[self.output_index["h_out"]].astype(np.float32)
            c_out = outputs[self.output_index["c_out"]].astype(np.float32)
        else:
            h_out = None
            c_out = None
        return actions.squeeze(0), h_out, c_out


class MujocoSoccerSim2Sim:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.policy = OnnxSoccerPolicy(args.onnx_path)
        self._xml_runtime_path = self._prepare_runtime_xml(args.xml_path)
        self.model = mujoco.MjModel.from_xml_path(self._xml_runtime_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = args.sim_dt

        self.isaac_joint_names = self.policy.metadata.joint_names
        self.body_names = self.policy.metadata.body_names
        self.anchor_body_name = self.policy.metadata.anchor_body_name
        self.default_joint_pos = self.policy.metadata.default_joint_pos
        self.joint_stiffness = self.policy.metadata.joint_stiffness
        self.joint_damping = self.policy.metadata.joint_damping
        self.action_scale = self.policy.metadata.action_scale
        self.observation_names = self.policy.metadata.observation_names
        self.motion_names = self.policy.metadata.motion_names
        self.motion_lengths = self.policy.reference.get("motion_lengths")
        self.multi_motion = self.policy.uses_motion_index
        self.final_anchor_positions = self._build_final_anchor_positions()

        self.isaac_to_mujoco_joint_index = self._build_joint_mapping()
        self.joint_qpos_addr = self._build_joint_addr_array(kind="qpos")
        self.joint_qvel_addr = self._build_joint_addr_array(kind="qvel")
        self.isaac_to_actuator_index = self._build_actuator_mapping()

        self.pelvis_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.ball_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, args.ball_body_name)
        self.ball_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, args.ball_body_name)
        if self.ball_body_id < 0 or self.ball_joint_id < 0:
            raise RuntimeError(f"Unable to locate ball body/joint named '{args.ball_body_name}' in MuJoCo XML.")
        self.goal_marker_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
        if self.goal_marker_body_id < 0:
            raise RuntimeError("Unable to locate goal_marker body in MuJoCo XML.")
        self.goal_marker_mocap_id = int(self.model.body_mocapid[self.goal_marker_body_id])
        if self.goal_marker_mocap_id < 0:
            raise RuntimeError("goal_marker body exists but is not a mocap body.")

        self.ball_qpos_addr = int(self.model.jnt_qposadr[self.ball_joint_id])
        self.ball_qvel_addr = int(self.model.jnt_dofadr[self.ball_joint_id])
        self.anchor_body_index = self.body_names.index(self.anchor_body_name)
        self.pelvis_body_index = self.body_names.index("pelvis")

        default_max_episode_steps = 500
        self.max_episode_steps = args.max_episode_steps if args.max_episode_steps > 0 else default_max_episode_steps

        self.h_state, self.c_state = self._zero_recurrent_state()
        self.last_action = np.zeros_like(self.default_joint_pos, dtype=np.float32)
        self.target_joint_pos = self.default_joint_pos.copy()
        self.time_step = 0
        self.current_motion_idx = 0
        self.current_motion_length = 0
        self.policy_step_count = 0
        self.kick_reset_countdown: int | None = None
        self.policy_active = False
        self.waiting_for_manual_start = False
        self.pending_start = False
        self.pending_reset = False
        self.pending_ball_delta = np.zeros(3, dtype=np.float32)
        self.goal_world = np.zeros(3, dtype=np.float32)
        self.ball_spawn_world = np.zeros(3, dtype=np.float32)
        self.initial_ball_planar_xy = np.zeros(2, dtype=np.float32)
        self.last_ball_speed_xy = 0.0
        self.current_ball_noise_sigma = 0.0
        self.current_goal_noise_sigma = 0.0
        self.current_ball_noise_vec = np.zeros(3, dtype=np.float32)
        self.current_goal_noise_vec = np.zeros(3, dtype=np.float32)

        self.reset()

    def _prepare_runtime_xml(self, xml_path: str) -> str:
        path = Path(xml_path)
        xml_text = path.read_text(encoding="utf-8")
        has_ball = f'name="{self.args.ball_body_name}"' in xml_text
        has_goal_marker = 'name="goal_marker"' in xml_text
        if has_ball and has_goal_marker:
            return str(path)

        ball_block = "" if has_ball else f"""
    <body name="{self.args.ball_body_name}" pos="1.0 0.0 {self.args.ball_height:.5f}" quat="1 0 0 0">
        <joint limited="false" name="{self.args.ball_body_name}" type="free"/>
        <geom name="{self.args.ball_body_name}_geom" type="sphere" size="{self.args.ball_radius:.5f}"
              contype="1" conaffinity="1" friction="0.8 0.01 0.0001" density="150"
              solref="0.02 1" solimp="0.9 0.95 0.01" rgba="1 1 1 1"/>
    </body>
"""
        goal_block = "" if has_goal_marker else f"""
    <body name="goal_marker" mocap="true" pos="0.0 0.0 {self.args.goal_center_z:.5f}">
        <geom name="goal_marker_geom" type="sphere" size="{self.args.goal_marker_radius:.5f}"
              contype="0" conaffinity="0" rgba="1 0 0 1"/>
    </body>
"""
        if "</worldbody>" not in xml_text:
            raise RuntimeError(f"MuJoCo XML at {xml_path} is missing </worldbody>; cannot inject ball body.")
        xml_text = xml_text.replace("</worldbody>", ball_block + goal_block + "\n</worldbody>", 1)

        runtime_path = path.with_name(f"{path.stem}_runtime_ball.xml")
        runtime_path.write_text(xml_text, encoding="utf-8")
        return str(runtime_path)

    def _build_joint_mapping(self) -> np.ndarray:
        mujoco_joint_names: list[str] = []
        for joint_id in range(self.model.njnt):
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            mujoco_joint_names.append(self.model.joint(joint_id).name)
        joint_name_to_mj_index = {name: idx for idx, name in enumerate(mujoco_joint_names)}
        try:
            return np.asarray([joint_name_to_mj_index[name] for name in self.isaac_joint_names], dtype=np.int32)
        except KeyError as exc:
            raise RuntimeError(f"MuJoCo XML is missing joint '{exc.args[0]}' required by the ONNX metadata.") from exc

    def _build_joint_addr_array(self, kind: str) -> np.ndarray:
        addrs = []
        for joint_name in self.isaac_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"Unable to resolve MuJoCo joint '{joint_name}'.")
            if kind == "qpos":
                addrs.append(int(self.model.jnt_qposadr[joint_id]))
            else:
                addrs.append(int(self.model.jnt_dofadr[joint_id]))
        return np.asarray(addrs, dtype=np.int32)

    def _build_actuator_mapping(self) -> np.ndarray:
        if self.model.nu == 0:
            raise RuntimeError(
                "The selected MuJoCo XML has no actuators (model.nu == 0). "
                "Use the actuator-enabled model, e.g. "
                "`source/whole_body_tracking/soccer/assets/unitree_description/mjcf/g1_actuator.xml`."
            )
        joint_to_actuator: dict[str, int] = {}
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            joint_name = self.model.joint(joint_id).name
            joint_to_actuator[joint_name] = actuator_id
        try:
            return np.asarray([joint_to_actuator[name] for name in self.isaac_joint_names], dtype=np.int32)
        except KeyError as exc:
            raise RuntimeError(f"MuJoCo actuators do not cover joint '{exc.args[0]}'.") from exc

    def _zero_recurrent_state(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not self.policy.is_recurrent:
            return None, None
        assert self.policy.recurrent_shape is not None
        num_layers, hidden_dim = self.policy.recurrent_shape
        return (
            np.zeros((num_layers, 1, hidden_dim), dtype=np.float32),
            np.zeros((num_layers, 1, hidden_dim), dtype=np.float32),
        )

    def _sample_goal_world(self) -> np.ndarray:
        dx = (np.random.rand() - 0.5) * self.args.goal_length
        dy = (np.random.rand() - 0.5) * self.args.goal_width
        return np.array(
            [
                self.args.goal_center_x + dx,
                self.args.goal_center_y + dy,
                self.args.goal_center_z,
            ],
            dtype=np.float32,
        )

    def _build_final_anchor_positions(self) -> np.ndarray:
        if self.policy.metadata.final_anchor_pos.size > 0:
            return self.policy.metadata.final_anchor_pos.astype(np.float32)
        if self.multi_motion:
            positions = []
            for motion_idx in range(len(self.motion_names)):
                motion_len = int(self.motion_lengths[motion_idx])
                positions.append(
                    self.policy.reference["body_pos_w"][motion_idx, motion_len - 1, self.anchor_body_index].astype(np.float32)
                )
            return np.stack(positions, axis=0)
        return self.policy.reference["body_pos_w"][-1, self.anchor_body_index][None, :].astype(np.float32)

    def _sample_ball_world(self) -> np.ndarray:
        if self.multi_motion:
            xy = self.final_anchor_positions[:, :2]
            x_min, y_min = np.min(xy, axis=0)
            x_max, y_max = np.max(xy, axis=0)
            margin = 0.20
            sampled_xy = np.array(
                [
                    np.random.uniform(x_min - margin, x_max + margin),
                    np.random.uniform(y_min - margin, y_max + margin),
                ],
                dtype=np.float32,
            )
            return np.array([sampled_xy[0], sampled_xy[1], self.args.ball_height], dtype=np.float32)

        body_pos = self.policy.reference["body_pos_w"]
        first_anchor = body_pos[0, self.anchor_body_index]
        last_anchor = body_pos[-1, self.anchor_body_index]
        radius_vec = last_anchor[:2] - first_anchor[:2]
        radius = float(np.linalg.norm(radius_vec))
        if radius > 1e-6:
            base_direction = radius_vec / radius
            base_angle = math.atan2(base_direction[1], base_direction[0])
        else:
            base_angle = 0.0
            radius = 0.0

        angle_offset = np.random.uniform(-self.args.ball_arc_angle, self.args.ball_arc_angle)
        direction = np.array([math.cos(base_angle + angle_offset), math.sin(base_angle + angle_offset)], dtype=np.float32)
        radius_offset = np.random.uniform(self.args.ball_radius_offset_min, self.args.ball_radius_offset_max)
        spawn_radius = max(0.0, radius + radius_offset)
        xy = first_anchor[:2] + spawn_radius * direction
        return np.array([xy[0], xy[1], self.args.ball_height], dtype=np.float32)

    def _get_joint_pos_isaac(self) -> np.ndarray:
        return self.data.qpos[self.joint_qpos_addr].astype(np.float32)

    def _get_joint_vel_isaac(self) -> np.ndarray:
        return self.data.qvel[self.joint_qvel_addr].astype(np.float32)

    def _get_pelvis_world(self) -> tuple[np.ndarray, np.ndarray]:
        return self.data.xpos[self.pelvis_body_id].astype(np.float32), self.data.xquat[self.pelvis_body_id].astype(np.float32)

    def _get_ball_world(self) -> tuple[np.ndarray, np.ndarray]:
        pos = self.data.qpos[self.ball_qpos_addr : self.ball_qpos_addr + 3].astype(np.float32)
        vel = self.data.qvel[self.ball_qvel_addr : self.ball_qvel_addr + 3].astype(np.float32)
        return pos, vel

    def _current_reference(self) -> dict[str, np.ndarray]:
        idx = min(self.time_step, self.current_motion_length - 1)
        if self.multi_motion:
            return {
                "joint_pos": self.policy.reference["joint_pos"][self.current_motion_idx, idx],
                "joint_vel": self.policy.reference["joint_vel"][self.current_motion_idx, idx],
                "body_ang_vel_w": self.policy.reference["body_ang_vel_w"][self.current_motion_idx, idx],
            }
        return {
            "joint_pos": self.policy.reference["joint_pos"][idx],
            "joint_vel": self.policy.reference["joint_vel"][idx],
            "body_ang_vel_w": self.policy.reference["body_ang_vel_w"][idx],
        }

    def _apply_soccer_observation_noise(
        self,
        ball_local: np.ndarray,
        goal_local: np.ndarray,
        ball_speed_xy: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.args.enable_soccer_noise:
            self.current_ball_noise_sigma = 0.0
            self.current_goal_noise_sigma = 0.0
            self.current_ball_noise_vec[:] = 0.0
            self.current_goal_noise_vec[:] = 0.0
            return ball_local, goal_local

        ball_dist = float(np.linalg.norm(ball_local))
        goal_dist = float(np.linalg.norm(goal_local))
        ball_sigma = (
            self.args.ball_noise_base_std
            + self.args.ball_noise_dist_coeff * ball_dist
            + self.args.ball_noise_vel_coeff * ball_speed_xy
        )
        goal_sigma = self.args.goal_noise_base_std + self.args.goal_noise_dist_coeff * goal_dist

        ball_noise = np.random.normal(0.0, ball_sigma, size=3).astype(np.float32)
        goal_noise = np.random.normal(0.0, goal_sigma, size=3).astype(np.float32)

        self.current_ball_noise_sigma = float(ball_sigma)
        self.current_goal_noise_sigma = float(goal_sigma)
        self.current_ball_noise_vec = ball_noise
        self.current_goal_noise_vec = goal_noise
        return ball_local + ball_noise, goal_local + goal_noise

    def _maybe_log_soccer_noise(self):
        if not self.args.enable_soccer_noise:
            return
        if self.policy_step_count == 0 or self.policy_step_count % max(1, self.args.soccer_noise_print_interval) != 0:
            return
        print(
            "[INFO] Soccer noise | "
            f"ball_sigma={self.current_ball_noise_sigma:.4f} "
            f"ball_noise={self.current_ball_noise_vec.tolist()} | "
            f"goal_sigma={self.current_goal_noise_sigma:.4f} "
            f"goal_noise={self.current_goal_noise_vec.tolist()}"
        )

    def _build_obs(self) -> np.ndarray:
        ref = self._current_reference()
        pelvis_pos_w, pelvis_quat_w = self._get_pelvis_world()
        ball_pos_w, ball_vel_w = self._get_ball_world()

        term_map: dict[str, np.ndarray] = {}
        term_map["command"] = np.concatenate((ref["joint_pos"], ref["joint_vel"]), axis=0).astype(np.float32)
        term_map["projected_gravity"] = quat_rotate_inverse_wxyz(pelvis_quat_w, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        term_map["motion_ref_ang_vel"] = ref["body_ang_vel_w"][self.anchor_body_index].astype(np.float32)
        term_map["base_ang_vel"] = self.data.qvel[3:6].astype(np.float32)
        term_map["joint_pos"] = (self._get_joint_pos_isaac() - self.default_joint_pos).astype(np.float32)
        term_map["joint_vel"] = self._get_joint_vel_isaac().astype(np.float32)
        term_map["actions"] = self.last_action.astype(np.float32)

        ball_local = quat_rotate_inverse_wxyz(pelvis_quat_w, ball_pos_w - pelvis_pos_w)
        goal_local = quat_rotate_inverse_wxyz(pelvis_quat_w, self.goal_world - pelvis_pos_w)
        ball_speed_xy = float(np.linalg.norm(ball_vel_w[:2]))
        ball_local, goal_local = self._apply_soccer_observation_noise(ball_local, goal_local, ball_speed_xy)
        term_map["target_point_pos"] = ball_local.astype(np.float32)
        term_map["target_destination_pos_local"] = goal_local.astype(np.float32)

        obs_terms: list[np.ndarray] = []
        missing = []
        for name in self.observation_names:
            term = term_map.get(name)
            if term is None:
                missing.append(name)
                continue
            obs_terms.append(term.reshape(-1))
        if missing:
            raise RuntimeError(f"Observation terms not implemented in MuJoCo bridge: {missing}")

        obs = np.concatenate(obs_terms, axis=0).astype(np.float32)
        if obs.shape[0] != self.policy.obs_dim:
            raise RuntimeError(f"Constructed obs dim {obs.shape[0]} does not match ONNX input dim {self.policy.obs_dim}.")
        return obs.reshape(1, -1)

    def _apply_action(self, action: np.ndarray):
        action = np.clip(action, -100.0, 100.0).astype(np.float32)
        target_joint_pos = self.default_joint_pos + self.action_scale * action
        self.target_joint_pos = target_joint_pos
        self.last_action = action.copy()

    def _compute_torque(self) -> np.ndarray:
        joint_pos = self._get_joint_pos_isaac()
        joint_vel = self._get_joint_vel_isaac()
        tau = (self.target_joint_pos - joint_pos) * self.joint_stiffness - joint_vel * self.joint_damping
        return tau.astype(np.float32)

    def _set_robot_state_from_reference(self):
        if self.multi_motion:
            root_pos = self.policy.reference["body_pos_w"][self.current_motion_idx, 0, self.pelvis_body_index]
            root_quat = self.policy.reference["body_quat_w"][self.current_motion_idx, 0, self.pelvis_body_index]
            joint_pos = self.policy.reference["joint_pos"][self.current_motion_idx, 0]
        else:
            root_pos = self.policy.reference["body_pos_w"][0, self.pelvis_body_index]
            root_quat = self.policy.reference["body_quat_w"][0, self.pelvis_body_index]
            joint_pos = self.policy.reference["joint_pos"][0]

        self.data.qpos[0:3] = root_pos
        self.data.qpos[3:7] = root_quat
        self.data.qpos[self.joint_qpos_addr] = joint_pos
        self.data.qvel[:] = 0.0
        self.target_joint_pos = joint_pos.astype(np.float32)

    def _set_ball_state(self, ball_world: np.ndarray):
        self.data.qpos[self.ball_qpos_addr : self.ball_qpos_addr + 3] = ball_world
        self.data.qpos[self.ball_qpos_addr + 3 : self.ball_qpos_addr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.data.qvel[self.ball_qvel_addr : self.ball_qvel_addr + 6] = 0.0

    def _set_goal_marker_state(self):
        self.data.mocap_pos[self.goal_marker_mocap_id] = self.goal_world.astype(np.float32)
        self.data.mocap_quat[self.goal_marker_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def _select_motion_for_ball(self, ball_world_xy: np.ndarray) -> int:
        selected = np.linalg.norm(self.final_anchor_positions[:, :2] - ball_world_xy[:2], axis=1).argmin()
        return int(selected)

    def _current_motion_name(self) -> str:
        if self.multi_motion and self.motion_names:
            return self.motion_names[self.current_motion_idx]
        return "single_motion"

    def _arm_policy_start(self, preserve_ball_pose: bool):
        if preserve_ball_pose:
            current_ball_world, _ = self._get_ball_world()
            self.ball_spawn_world = current_ball_world.copy()
        self.current_motion_idx = self._select_motion_for_ball(self.ball_spawn_world)
        self.current_motion_length = (
            int(self.motion_lengths[self.current_motion_idx]) if self.multi_motion else int(self.policy.reference["time_step_total"])
        )
        self.time_step = 0
        self.policy_step_count = 0
        self.kick_reset_countdown = None
        self.h_state, self.c_state = self._zero_recurrent_state()
        self.last_action[:] = 0.0
        self.initial_ball_planar_xy = self.ball_spawn_world[:2].copy()
        mujoco.mj_resetData(self.model, self.data)
        self._set_robot_state_from_reference()
        self._set_ball_state(self.ball_spawn_world)
        self._set_goal_marker_state()
        mujoco.mj_forward(self.model, self.data)

    def _enter_waiting_state(self):
        self.policy_active = False
        self.waiting_for_manual_start = True
        print(
            "[INFO] Episode reset. Adjust ball with WASD (x/y) and Q/E (z), then press SPACE/ENTER to start policy. "
            "Press R to resample/reset."
        )

    def _move_ball_manual(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
        if self.policy_active:
            return
        ball_world, _ = self._get_ball_world()
        ball_world = ball_world.copy()
        ball_world[0] += dx
        ball_world[1] += dy
        ball_world[2] = max(self.args.ball_radius, ball_world[2] + dz)
        self.ball_spawn_world = ball_world
        self._set_ball_state(ball_world)
        mujoco.mj_forward(self.model, self.data)
        print(f"[INFO] Ball moved to {ball_world.tolist()}")

    def _flush_pending_ui_actions(self):
        if self.pending_reset:
            self.pending_reset = False
            self.pending_start = False
            self.pending_ball_delta[:] = 0.0
            self.reset()
            return

        if np.any(self.pending_ball_delta != 0.0):
            dx, dy, dz = self.pending_ball_delta.tolist()
            self.pending_ball_delta[:] = 0.0
            self._move_ball_manual(dx=dx, dy=dy, dz=dz)

        if self.pending_start:
            self.pending_start = False
            if not self.policy_active:
                self._start_policy()

    def _start_policy(self):
        self._arm_policy_start(preserve_ball_pose=True)
        self.policy_active = True
        self.waiting_for_manual_start = False
        print(
            f"[INFO] Starting policy with motion_idx={self.current_motion_idx}, "
            f"motion_name={self._current_motion_name()}, goal={self.goal_world.tolist()}, "
            f"ball={self.ball_spawn_world.tolist()}"
        )

    def _on_key(self, keycode: int):
        if keycode in (SPACE_KEY, ENTER_KEY):
            if not self.policy_active:
                self.pending_start = True
        elif keycode in (R_KEY_LOWER, R_KEY_UPPER):
            self.pending_reset = True
        elif keycode in (W_KEY_LOWER, W_KEY_UPPER):
            self.pending_ball_delta[0] += self.args.manual_ball_step
        elif keycode in (S_KEY_LOWER, S_KEY_UPPER):
            self.pending_ball_delta[0] -= self.args.manual_ball_step
        elif keycode in (A_KEY_LOWER, A_KEY_UPPER):
            self.pending_ball_delta[1] += self.args.manual_ball_step
        elif keycode in (D_KEY_LOWER, D_KEY_UPPER):
            self.pending_ball_delta[1] -= self.args.manual_ball_step
        elif keycode in (Q_KEY_LOWER, Q_KEY_UPPER):
            self.pending_ball_delta[2] += self.args.manual_ball_step
        elif keycode in (E_KEY_LOWER, E_KEY_UPPER):
            self.pending_ball_delta[2] -= self.args.manual_ball_step

    def reset(self):
        self.goal_world = self._sample_goal_world()
        self.ball_spawn_world = self._sample_ball_world()
        self.last_ball_speed_xy = 0.0
        self._arm_policy_start(preserve_ball_pose=False)
        self._enter_waiting_state()

    def _should_reset(self) -> bool:
        pelvis_pos_w, _ = self._get_pelvis_world()
        ball_pos_w, ball_vel_w = self._get_ball_world()
        ball_speed_xy = float(np.linalg.norm(ball_vel_w[:2]))
        self.last_ball_speed_xy = ball_speed_xy

        if pelvis_pos_w[2] < self.args.fall_height_threshold:
            return True

        if self.policy_step_count >= self.max_episode_steps:
            return True

        if self.kick_reset_countdown is None:
            ball_displacement = np.linalg.norm(ball_pos_w[:2] - self.initial_ball_planar_xy)
            if (
                self.time_step > self.args.kick_armed_min_steps
                and ball_speed_xy > self.args.kick_speed_threshold
                and ball_displacement > 0.05
            ):
                self.kick_reset_countdown = self.args.kick_reset_delay_steps
        else:
            self.kick_reset_countdown -= 1
            if self.kick_reset_countdown <= 0:
                return True

        return False

    def run(self):
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=self._on_key) as viewer:
            start = time.time()
            while viewer.is_running() and time.time() - start < self.args.simulation_duration:
                step_start = time.time()
                self._flush_pending_ui_actions()
                if self.policy_active and self._should_reset():
                    self.reset()
                    continue

                if self.policy_active and (
                    self.data.time == 0.0 or int(self.data.time / self.args.sim_dt) % self.args.control_decimation == 0
                ):
                    obs = self._build_obs()
                    action, h_out, c_out = self.policy.act(
                        obs, self.time_step, self.current_motion_idx, self.h_state, self.c_state
                    )
                    self._apply_action(action)
                    self.h_state, self.c_state = h_out, c_out
                    self.time_step += 1
                    self.policy_step_count += 1
                    self._maybe_log_soccer_noise()
                elif not self.policy_active:
                    self.data.ctrl[:] = 0.0

                if self.policy_active:
                    tau = self._compute_torque()
                    ctrl = np.zeros(self.model.nu, dtype=np.float32)
                    ctrl[self.isaac_to_actuator_index] = tau
                    self.data.ctrl[:] = ctrl
                    mujoco.mj_step(self.model, self.data)
                else:
                    mujoco.mj_forward(self.model, self.data)
                viewer.sync()

                elapsed = time.time() - step_start
                sleep_time = self.args.sim_dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)


def main():
    args = parse_args()
    runner = MujocoSoccerSim2Sim(args)
    runner.run()


if __name__ == "__main__":
    main()
