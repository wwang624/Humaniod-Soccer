"""Convert video2robot Saya/v11 robot_motion.pkl to HumanoidSoccer motion npz."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert video2robot Saya robot_motion.pkl to motion npz.")
parser.add_argument("--input", required=True, type=Path, help="Path to video2robot robot_motion.pkl.")
parser.add_argument("--output", required=True, type=Path, help="Output motion .npz.")
parser.add_argument("--input_fps", type=float, default=None, help="Override input FPS. Defaults to pkl['fps'].")
parser.add_argument("--output_fps", type=int, default=50, help="Output FPS.")
parser.add_argument("--kick_leg", choices=["left", "right", "unknown"], default="right", help="Kick-leg metadata.")
parser.add_argument("--recentre_xy", action="store_true", help="Subtract first root XY before replay.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from soccer.robots.saya_29dof import SAYA_29DOF_CFG, SAYA_29DOF_SOURCE_JOINT_NAMES


class NumpyCompatUnpickler(pickle.Unpickler):
    """Load NumPy-2 pickles in NumPy-1 environments used by IsaacLab."""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


SAYA_VIDEO2ROBOT_JOINT_NAMES = (
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


@configclass
class SayaReplaySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = SAYA_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


class SayaVideo2RobotMotionLoader:
    def __init__(self, path: Path, input_fps: float | None, output_fps: int, device: torch.device, recentre_xy: bool):
        self.path = path
        self.device = device
        self.output_fps = float(output_fps)
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self._load(input_fps, recentre_xy)
        self._interpolate()
        self._compute_velocities()

    def _load_motion_dict(self) -> dict[str, np.ndarray | float | str]:
        if self.path.suffix == ".npz":
            data = np.load(self.path, allow_pickle=True)
            return {key: data[key] for key in data.files}
        with open(self.path, "rb") as f:
            return NumpyCompatUnpickler(f).load()

    def _load(self, input_fps: float | None, recentre_xy: bool) -> None:
        data = self._load_motion_dict()

        root_pos = np.asarray(data["root_pos"], dtype=np.float32)
        root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float32)
        dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            raise ValueError(f"root_pos must be (T, 3), got {root_pos.shape}")
        if root_rot_xyzw.ndim != 2 or root_rot_xyzw.shape[1] != 4:
            raise ValueError(f"root_rot must be (T, 4), got {root_rot_xyzw.shape}")
        if dof_pos.ndim != 2 or dof_pos.shape[1] != len(SAYA_VIDEO2ROBOT_JOINT_NAMES):
            raise ValueError(f"dof_pos must be (T, {len(SAYA_VIDEO2ROBOT_JOINT_NAMES)}), got {dof_pos.shape}")

        robot_type_raw = data.get("robot_type", "unknown")
        if isinstance(robot_type_raw, np.ndarray):
            robot_type_raw = robot_type_raw.item()
        robot_type = str(robot_type_raw)
        if robot_type not in {"v11", "w11", "whole_self_v11", "saya", "saya_29dof"}:
            print(f"[WARN] Unexpected robot_type={robot_type}; continuing with Saya 29DoF joint order.")

        source_index = {name: idx for idx, name in enumerate(SAYA_VIDEO2ROBOT_JOINT_NAMES)}
        reorder = [source_index[name] for name in SAYA_29DOF_SOURCE_JOINT_NAMES]
        dof_pos = dof_pos[:, reorder]

        if recentre_xy:
            root_pos = root_pos.copy()
            root_pos[:, :2] -= root_pos[0:1, :2]

        root_rot_xyzw = root_rot_xyzw / np.clip(np.linalg.norm(root_rot_xyzw, axis=1, keepdims=True), 1.0e-8, None)
        root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]

        fps_raw = data.get("fps", 30.0)
        if isinstance(fps_raw, np.ndarray):
            fps_raw = fps_raw.item()
        self.input_fps = float(input_fps if input_fps is not None else fps_raw)
        self.input_dt = 1.0 / self.input_fps
        self.input_frames = int(dof_pos.shape[0])
        self.duration = (self.input_frames - 1) * self.input_dt
        self.root_pos_input = torch.as_tensor(root_pos, dtype=torch.float32, device=self.device)
        self.root_rot_input = torch.as_tensor(root_rot_wxyz, dtype=torch.float32, device=self.device)
        self.dof_pos_input = torch.as_tensor(dof_pos, dtype=torch.float32, device=self.device)
        print(
            f"[INFO] Loaded {self.path}: robot_type={robot_type} frames={self.input_frames} "
            f"fps={self.input_fps:g} duration={self.duration:.3f}s"
        )

    def _interpolate(self) -> None:
        times = torch.arange(0.0, self.duration + 1.0e-6, self.output_dt, device=self.device)
        self.output_frames = int(times.shape[0])
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.root_pos = self._lerp(self.root_pos_input[index_0], self.root_pos_input[index_1], blend.unsqueeze(1))
        self.root_rot = self._slerp(self.root_rot_input[index_0], self.root_rot_input[index_1], blend)
        self.dof_pos = self._lerp(self.dof_pos_input[index_0], self.dof_pos_input[index_1], blend.unsqueeze(1))
        print(f"[INFO] Interpolated to {self.output_frames} frames at {self.output_fps:g} Hz")

    def _compute_frame_blend(self, times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1, device=self.device))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    @staticmethod
    def _lerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1.0 - blend) + b * blend

    @staticmethod
    def _slerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(a)
        for idx in range(a.shape[0]):
            out[idx] = quat_slerp(a[idx], b[idx], blend[idx])
        return out

    def _compute_velocities(self) -> None:
        self.root_lin_vel = torch.gradient(self.root_pos, spacing=self.output_dt, dim=0)[0]
        self.dof_vel = torch.gradient(self.dof_pos, spacing=self.output_dt, dim=0)[0]
        self.root_ang_vel = self._so3_derivative(self.root_rot, self.output_dt)

    @staticmethod
    def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
        if rotations.shape[0] < 3:
            return torch.zeros((rotations.shape[0], 3), dtype=rotations.dtype, device=rotations.device)
        q_rel = quat_mul(rotations[2:], quat_conjugate(rotations[:-2]))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self):
        state = (
            self.root_pos[self.current_idx : self.current_idx + 1],
            self.root_rot[self.current_idx : self.current_idx + 1],
            self.root_lin_vel[self.current_idx : self.current_idx + 1],
            self.root_ang_vel[self.current_idx : self.current_idx + 1],
            self.dof_pos[self.current_idx : self.current_idx + 1],
            self.dof_vel[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        return state, self.current_idx >= self.output_frames


def run_converter(sim: SimulationContext, scene: InteractiveScene) -> None:
    motion = SayaVideo2RobotMotionLoader(
        args_cli.input,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        recentre_xy=args_cli.recentre_xy,
    )
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(list(SAYA_29DOF_SOURCE_JOINT_NAMES), preserve_order=True)[0]

    log: dict[str, list[np.ndarray] | np.ndarray] = {
        "fps": np.array([args_cli.output_fps], dtype=np.int64),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
        "kick_leg": np.array(args_cli.kick_leg),
        "joint_names": np.asarray(robot.joint_names, dtype=str),
        "body_names": np.asarray(robot.body_names, dtype=str),
    }

    frame_dt = 1.0 / float(args_cli.output_fps)
    while simulation_app.is_running():
        frame_start = time.perf_counter()
        (root_pos, root_rot, root_lin_vel, root_ang_vel, dof_pos, dof_vel), done = motion.get_next_state()

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = root_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = root_rot
        root_states[:, 7:10] = root_lin_vel
        root_states[:, 10:] = root_ang_vel
        robot.write_root_state_to_sim(root_states)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = dof_pos
        joint_vel[:, robot_joint_indexes] = dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)

        sim.render()
        scene.update(sim.get_physics_dt())

        log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
        log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
        log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
        log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
        log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
        log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if done:
            break
        elapsed = time.perf_counter() - frame_start
        if elapsed < frame_dt:
            time.sleep(frame_dt - elapsed)

    output = {
        key: np.stack(value, axis=0).astype(np.float32) if isinstance(value, list) else value
        for key, value in log.items()
    }
    args_cli.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args_cli.output, **output)
    print(f"[INFO] Saved {args_cli.output}")
    for key, value in output.items():
        print(f"  {key}: shape={getattr(value, 'shape', None)} dtype={getattr(value, 'dtype', None)}")


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / float(args_cli.output_fps)
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(SayaReplaySceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    run_converter(sim, scene)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
