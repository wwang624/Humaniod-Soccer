# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch
import json

import onnx

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from soccer.tasks.tracking.mdp import MotionCommand


def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
    motion_name=None
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose, motion_name)
    policy_exporter.export(path, filename)


def export_multi_motion_policy_as_onnx(
    env: ManagerBasedRLEnv,
    actor_critic: object,
    path: str,
    normalizer: object | None = None,
    filename="policy_bundle.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxMultiMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False, motion_name=None):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")
        # import ipdb; ipdb.set_trace()
        if len(cmd.motion.joint_pos.shape) == 2:  # Single motion.
            self.joint_pos = cmd.motion.joint_pos.to("cpu")
            self.joint_vel = cmd.motion.joint_vel.to("cpu")
            self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
            self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
            self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
            self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
            self.time_step_total = self.joint_pos.shape[0]
        elif len(cmd.motion.joint_pos.shape) == 3:  # Multi-motion.
            # Strip extension to match motion_name entries (stored without suffix).
            motion_name_no_ext = motion_name.split(".")[0] if motion_name else motion_name
            idx = cmd.motion.motion_name.index(motion_name_no_ext)
            self.joint_pos = cmd.motion.joint_pos[idx][:cmd.motion.motion_lengths[idx]].to("cpu")
            self.joint_vel = cmd.motion.joint_vel[idx][:cmd.motion.motion_lengths[idx]].to("cpu")
            self.body_pos_w = cmd.motion.body_pos_w[idx][:cmd.motion.motion_lengths[idx]].to("cpu")
            self.body_quat_w = cmd.motion.body_quat_w[idx][:cmd.motion.motion_lengths[idx]].to("cpu")
            self.body_lin_vel_w = cmd.motion.body_lin_vel_w[idx][:cmd.motion.motion_lengths[idx]].to("cpu")
            self.body_ang_vel_w = cmd.motion.body_ang_vel_w[idx][:cmd.motion.motion_lengths[idx]].to("cpu")
            self.time_step_total = self.joint_pos.shape[0]           


    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(x)),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
            self.time_step_total * torch.ones_like(time_step_clamped, dtype=torch.float32).unsqueeze(-1),
        )

    def forward_lstm(self, x_in, h_in, c_in, time_step):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(x),
            h,
            c,
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
            self.time_step_total * torch.ones_like(time_step_clamped, dtype=torch.float32).unsqueeze(-1),
        )

    def export(self, path, filename):
        self.to("cpu")
        self.eval()

        if self.is_recurrent:
            class _LstmExportWrapper(torch.nn.Module):
                def __init__(self, parent):
                    super().__init__()
                    self.parent = parent

                def forward(self, obs, h_in, c_in, time_step):
                    return self.parent.forward_lstm(obs, h_in, c_in, time_step)

            wrapper = _LstmExportWrapper(self)

            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            time_step = torch.zeros(1, 1)
            torch.onnx.export(
                wrapper,
                (obs, h_in, c_in, time_step),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in", "time_step"],
                output_names=[
                    "actions",
                    "h_out",
                    "c_out",
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                    "time_step_total",
                ],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            time_step = torch.zeros(1, 1)
            torch.onnx.export(
                self,
                (obs, time_step),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "time_step"],
                output_names=[
                    "actions",
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                ],
                dynamic_axes={},
            )


class _OnnxMultiMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")
        if len(cmd.motion.joint_pos.shape) != 3:
            raise ValueError("Bundled multi-motion export requires a multi-motion command tensor.")

        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.motion_lengths = cmd.motion.file_lengths.to("cpu")
        self.motion_count = int(self.joint_pos.shape[0])
        self.max_time_step_total = int(self.joint_pos.shape[1])
        if self.is_recurrent:
            self.forward = self.forward_lstm
        else:
            self.forward = self.forward_feedforward

    def _select_reference(self, motion_idx: torch.Tensor, time_step: torch.Tensor):
        motion_idx = torch.clamp(motion_idx.long().reshape(-1), min=0, max=self.motion_count - 1)
        selected_lengths = self.motion_lengths[motion_idx]
        time_step = torch.clamp(time_step.long().reshape(-1), min=0)
        time_step = torch.minimum(time_step, selected_lengths - 1)
        return (
            motion_idx,
            time_step,
            selected_lengths,
            self.joint_pos[motion_idx, time_step],
            self.joint_vel[motion_idx, time_step],
            self.body_pos_w[motion_idx, time_step],
            self.body_quat_w[motion_idx, time_step],
            self.body_lin_vel_w[motion_idx, time_step],
            self.body_ang_vel_w[motion_idx, time_step],
        )

    def forward_feedforward(self, x, motion_idx, time_step):
        (
            motion_idx,
            time_step,
            selected_lengths,
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
        ) = self._select_reference(motion_idx, time_step)

        return (
            self.actor(self.normalizer(x)),
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
            selected_lengths.to(dtype=torch.float32).unsqueeze(-1),
            motion_idx.to(dtype=torch.float32).unsqueeze(-1),
            torch.full_like(selected_lengths, self.max_time_step_total, dtype=torch.float32).unsqueeze(-1),
        )

    def forward_lstm(self, x_in, h_in, c_in, motion_idx, time_step):
        x_in = self.normalizer(x_in)
        x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
        x = x.squeeze(0)
        (
            motion_idx,
            time_step,
            selected_lengths,
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
        ) = self._select_reference(motion_idx, time_step)

        return (
            self.actor(x),
            h,
            c,
            joint_pos,
            joint_vel,
            body_pos_w,
            body_quat_w,
            body_lin_vel_w,
            body_ang_vel_w,
            selected_lengths.to(dtype=torch.float32).unsqueeze(-1),
            motion_idx.to(dtype=torch.float32).unsqueeze(-1),
            torch.full_like(selected_lengths, self.max_time_step_total, dtype=torch.float32).unsqueeze(-1),
        )

    def export(self, path, filename):
        self.to("cpu")
        self.eval()

        if self.is_recurrent:
            class _LstmExportWrapper(torch.nn.Module):
                def __init__(self, parent):
                    super().__init__()
                    self.actor = parent.actor
                    self.rnn = parent.rnn
                    self.normalizer = parent.normalizer
                    self.joint_pos = parent.joint_pos
                    self.joint_vel = parent.joint_vel
                    self.body_pos_w = parent.body_pos_w
                    self.body_quat_w = parent.body_quat_w
                    self.body_lin_vel_w = parent.body_lin_vel_w
                    self.body_ang_vel_w = parent.body_ang_vel_w
                    self.motion_lengths = parent.motion_lengths
                    self.motion_count = parent.motion_count
                    self.max_time_step_total = parent.max_time_step_total

                def _select_reference(self, motion_idx, time_step):
                    motion_idx = torch.clamp(motion_idx.long().reshape(-1), min=0, max=self.motion_count - 1)
                    selected_lengths = self.motion_lengths[motion_idx]
                    time_step = torch.clamp(time_step.long().reshape(-1), min=0)
                    time_step = torch.minimum(time_step, selected_lengths - 1)
                    return (
                        motion_idx,
                        time_step,
                        selected_lengths,
                        self.joint_pos[motion_idx, time_step],
                        self.joint_vel[motion_idx, time_step],
                        self.body_pos_w[motion_idx, time_step],
                        self.body_quat_w[motion_idx, time_step],
                        self.body_lin_vel_w[motion_idx, time_step],
                        self.body_ang_vel_w[motion_idx, time_step],
                    )

                def forward(self, obs, h_in, c_in, motion_idx, time_step):
                    x_in = self.normalizer(obs)
                    x, (h, c) = self.rnn(x_in.unsqueeze(0), (h_in, c_in))
                    x = x.squeeze(0)
                    (
                        motion_idx,
                        _time_step,
                        selected_lengths,
                        joint_pos,
                        joint_vel,
                        body_pos_w,
                        body_quat_w,
                        body_lin_vel_w,
                        body_ang_vel_w,
                    ) = self._select_reference(motion_idx, time_step)
                    return (
                        self.actor(x),
                        h,
                        c,
                        joint_pos,
                        joint_vel,
                        body_pos_w,
                        body_quat_w,
                        body_lin_vel_w,
                        body_ang_vel_w,
                        selected_lengths.to(dtype=torch.float32).unsqueeze(-1),
                        motion_idx.to(dtype=torch.float32).unsqueeze(-1),
                        torch.full_like(selected_lengths, self.max_time_step_total, dtype=torch.float32).unsqueeze(-1),
                    )

            wrapper = _LstmExportWrapper(self)

            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            motion_idx = torch.zeros(1, 1)
            time_step = torch.zeros(1, 1)
            torch.onnx.export(
                wrapper,
                (obs, h_in, c_in, motion_idx, time_step),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in", "motion_idx", "time_step"],
                output_names=[
                    "actions",
                    "h_out",
                    "c_out",
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                    "motion_length",
                    "motion_idx_selected",
                    "time_step_total",
                ],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            motion_idx = torch.zeros(1, 1)
            time_step = torch.zeros(1, 1)
            torch.onnx.export(
                self,
                (obs, motion_idx, time_step),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "motion_idx", "time_step"],
                output_names=[
                    "actions",
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                    "motion_length",
                    "motion_idx_selected",
                    "time_step_total",
                ],
                dynamic_axes={},
            )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x) for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    onnx_path = os.path.join(path, filename)
    cmd: MotionCommand = env.command_manager.get_term("motion")
    metadata = {
        "run_path": run_path,
        "joint_names": env.scene["robot"].data.joint_names,
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(),
        "command_names": env.command_manager.active_terms,
        "observation_names": env.observation_manager.active_terms["policy"],
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(),
        "anchor_body_name": cmd.cfg.anchor_body_name,
        "soccer_obs_body_name": getattr(cmd, "soccer_obs_body_name", ""),
        "body_names": cmd.cfg.body_names,
        "root_body_name": next((name for name in ("pelvis", "base_link") if name in cmd.cfg.body_names), cmd.cfg.anchor_body_name),
    }
    if hasattr(cmd.motion, "motion_name"):
        metadata["motion_names"] = list(cmd.motion.motion_name)
    if hasattr(cmd.motion, "motion_lengths"):
        metadata["motion_lengths"] = list(cmd.motion.motion_lengths)
    if hasattr(cmd, "motion_kick_leg_names"):
        metadata["motion_kick_leg_names"] = list(cmd.motion_kick_leg_names)
    if hasattr(cmd.motion, "body_pos_w"):
        anchor_idx = cmd.motion_anchor_body_index
        if len(cmd.motion.body_pos_w.shape) == 4:
            final_anchor_pos = []
            for motion_idx, motion_len in enumerate(cmd.motion.motion_lengths):
                final_anchor_pos.append(
                    cmd.motion.body_pos_w[motion_idx, motion_len - 1, anchor_idx].detach().cpu().tolist()
                )
            metadata["final_anchor_pos"] = final_anchor_pos
        elif len(cmd.motion.body_pos_w.shape) == 3:
            metadata["final_anchor_pos"] = [
                cmd.motion.body_pos_w[cmd.motion.time_step_total - 1, anchor_idx].detach().cpu().tolist()
            ]

    model = onnx.load(onnx_path)

    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        if isinstance(v, list) and all(isinstance(item, (int, float)) for item in v):
            entry.value = list_to_csv_str(v)
        elif isinstance(v, list):
            entry.value = json.dumps(v)
        else:
            entry.value = str(v)
        model.metadata_props.append(entry)

    onnx.save(model, onnx_path)
