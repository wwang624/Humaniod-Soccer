from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.observations import base_ang_vel, joint_pos_rel, joint_vel_rel, last_action, projected_gravity
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
    yaw_quat,
)

from soccer.tasks.tracking.mdp.commands_multi_motion_soccer import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

def motion_anchor_ang_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.anchor_ang_vel_w.view(env.num_envs, -1)


def motion_phase_sin_cos(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    time_steps = command.time_steps.to(dtype=torch.float32)
    motion_length = getattr(command, "motion_length", None)
    if motion_length is None:
        total = float(getattr(command.motion, "time_step_total", 1))
        denom = torch.full_like(time_steps, max(total - 1.0, 1.0))
    else:
        denom = torch.clamp(motion_length.to(dtype=torch.float32) - 1.0, min=1.0)
    phase = torch.clamp(time_steps / denom, 0.0, 1.0)
    angle = 2.0 * math.pi * phase
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


def policy_obs_history(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    history_length: int = 4,
    use_noisy_targets: bool = False,
    ball_noise_std: tuple[float, float, float] = (0.03, 0.03, 0.02),
    goal_noise_std: tuple[float, float, float] = (0.03, 0.03, 0.02),
    noise_type: str = "normal",
    update_interval: int = 2,
    dropout_prob: float = 0.10,
    hold_last: bool = True,
) -> torch.Tensor:
    """Stack recent deployable proprioceptive and soccer observations for MLP students."""
    history_length = max(int(history_length), 1)
    if use_noisy_targets:
        ball_local, goal_local = _noisy_soccer_target_pair(
            env,
            command_name,
            ball_noise_std=ball_noise_std,
            goal_noise_std=goal_noise_std,
            noise_type=noise_type,
            update_interval=update_interval,
            dropout_prob=dropout_prob,
            hold_last=hold_last,
        )
    else:
        ball_local = constant_target_point_pos(env, command_name)
        goal_local = target_destination_pos_local(env, command_name)
    current = torch.cat(
        [
            projected_gravity(env),
            base_ang_vel(env),
            joint_pos_rel(env),
            joint_vel_rel(env),
            last_action(env),
            ball_local,
            goal_local,
        ],
        dim=-1,
    )

    cache_name = f"_{command_name}_policy_obs_history_cache"
    cache = getattr(env, cache_name, None)
    if cache is None or cache.shape[0] != env.num_envs or cache.shape[1] != history_length or cache.shape[2] != current.shape[-1]:
        cache = current.unsqueeze(1).repeat(1, history_length, 1)
        setattr(env, cache_name, cache)

    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is not None:
        reset_mask = step_buf.to(device=current.device) == 0
        if torch.any(reset_mask):
            cache[reset_mask] = current[reset_mask].unsqueeze(1).repeat(1, history_length, 1)

    cache = torch.roll(cache, shifts=-1, dims=1)
    cache[:, -1] = current
    setattr(env, cache_name, cache)
    return cache.reshape(env.num_envs, -1)


def _get_motion_command(env: ManagerBasedEnv, command_name: str) -> MotionCommand:
    command: MotionCommand | None = env.command_manager.get_term(command_name)
    if command is None:
        raise RuntimeError(f"motion command '{command_name}' not found in env.command_manager")
    if not hasattr(command, "target_point_pos"):
        raise RuntimeError(f"motion command '{command_name}' lacks target_point_pos attribute")
    return command


def get_target_point_world(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    target_local = command.target_point_pos
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        return target_local + env_origins
    return target_local


def get_target_point_base(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    target_world = get_target_point_world(env, command_name)
    delta = target_world - command.robot_soccer_obs_pos_w
    return quat_apply(quat_inv(command.robot_soccer_obs_quat_w), delta)


def _positional_encoding(vec: torch.Tensor, num_freqs: int = 6) -> torch.Tensor:
    """Apply sinusoidal positional encoding to a target tensor of shape (E, 3).

    The encoding follows Transformer-style frequencies: for each coordinate x,
    compute sin(2^k*pi*x) and cos(2^k*pi*x) for k=0..num_freqs-1, then
    concatenate with the original coordinates.
    """
    if num_freqs <= 0:
        return vec.view(vec.shape[0], -1)

    device = vec.device
    dtype = vec.dtype
    # freqs: [num_freqs]
    freqs = (2.0 ** torch.arange(num_freqs, device=device, dtype=dtype)) * math.pi
    # vec: [E, 3] -> vec_exp: [E, 3, num_freqs]
    vec_exp = vec.unsqueeze(-1) * freqs
    sin = torch.sin(vec_exp)
    cos = torch.cos(vec_exp)
    # sin_cos: [E, 3, 2*num_freqs] -> flatten per-sample
    sin_cos = torch.cat([sin, cos], dim=-1).view(vec.shape[0], -1)
    # Concatenate original coordinates in front.
    return torch.cat([vec.view(vec.shape[0], -1), sin_cos], dim=-1)


def target_point_pos_first_frame(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    cache_name = f"_{command_name}_target_point_cache"
    target_local = get_target_point_base(env, command_name)

    cache = getattr(env, cache_name, None)
    if cache is None or cache.shape[0] != env.num_envs:
        cache = target_local.clone()
        setattr(env, cache_name, cache)

    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is None:
        return cache

    first_step_mask = (step_buf == 0)
    if torch.any(first_step_mask):
        cache = getattr(env, cache_name)
        # Only refresh the cache when an environment just reset so the policy keeps the first-frame cue.
        cache[first_step_mask] = target_local[first_step_mask]
        setattr(env, cache_name, cache)
    # Return cached target vector.
    return getattr(env, cache_name)
    return _positional_encoding(getattr(env, cache_name), num_freqs=6)


def constant_target_point_pos(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    # Constant observation path keeps the same representation as policy inputs.
    base = get_target_point_base(env, command_name)
    return base
    return _positional_encoding(base, num_freqs=6)


def _sample_target_noise(
    clean: torch.Tensor,
    noise_type: str,
    noise_std: tuple[float, float, float],
) -> torch.Tensor:
    std = torch.as_tensor(noise_std, dtype=clean.dtype, device=clean.device).view(1, 3)
    if torch.all(std <= 0):
        return torch.zeros_like(clean)
    if noise_type == "uniform":
        return (torch.rand_like(clean) * 2.0 - 1.0) * std
    if noise_type == "normal":
        return torch.randn_like(clean) * std
    raise ValueError(f"Unsupported soccer target noise_type: {noise_type}")


def _noisy_soccer_target_pair(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    *,
    ball_noise_std: tuple[float, float, float] = (0.03, 0.03, 0.02),
    goal_noise_std: tuple[float, float, float] = (0.03, 0.03, 0.02),
    noise_type: str = "normal",
    update_interval: int = 2,
    dropout_prob: float = 0.10,
    hold_last: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return student perception targets with shared noise/dropout state.

    The ball and goal observations use one cache so an artificial perception dropout
    cannot update one target without the other. This keeps the observation contract
    close to deployment: valid detections refresh the target, missed detections hold
    the last valid target.
    """
    clean_ball = constant_target_point_pos(env, command_name)
    clean_goal = target_destination_pos_local(env, command_name)
    update_interval = max(int(update_interval), 1)
    dropout_prob = float(min(max(dropout_prob, 0.0), 1.0))

    cache_name = f"_{command_name}_noisy_soccer_target_cache"
    cache = getattr(env, cache_name, None)
    if cache is None or "step" not in cache or cache["ball"].shape[0] != env.num_envs:
        cache = {
            "ball": clean_ball.clone(),
            "goal": clean_goal.clone(),
            "step": torch.full((env.num_envs,), -1, dtype=torch.long, device=clean_ball.device),
        }
        setattr(env, cache_name, cache)

    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is None:
        update_mask = torch.ones(env.num_envs, dtype=torch.bool, device=clean_ball.device)
        first_step_mask = update_mask
        common_step = int(getattr(env, "common_step_counter", 0))
        current_step = torch.full((env.num_envs,), common_step, dtype=torch.long, device=clean_ball.device)
    else:
        current_step = step_buf.to(device=clean_ball.device, dtype=torch.long)
        already_updated_mask = cache["step"] == current_step
        if torch.all(already_updated_mask):
            return cache["ball"], cache["goal"]
        update_mask = (step_buf % update_interval) == 0
        first_step_mask = step_buf == 0
        update_mask = update_mask | first_step_mask
        update_mask = update_mask & ~already_updated_mask

    if not torch.any(update_mask):
        return cache["ball"], cache["goal"]

    valid_update_mask = update_mask.clone()
    if dropout_prob > 0.0:
        detected = torch.rand(env.num_envs, device=clean_ball.device) >= dropout_prob
        valid_update_mask = update_mask & (detected | first_step_mask)

    if not hold_last:
        invalid_update_mask = update_mask & ~valid_update_mask
        if torch.any(invalid_update_mask):
            cache["ball"][invalid_update_mask] = clean_ball[invalid_update_mask]
            cache["goal"][invalid_update_mask] = clean_goal[invalid_update_mask]

    if torch.any(valid_update_mask):
        noisy_ball = clean_ball + _sample_target_noise(clean_ball, noise_type, ball_noise_std)
        noisy_goal = clean_goal + _sample_target_noise(clean_goal, noise_type, goal_noise_std)
        cache["ball"][valid_update_mask] = noisy_ball[valid_update_mask]
        cache["goal"][valid_update_mask] = noisy_goal[valid_update_mask]

    cache["step"][update_mask] = current_step[update_mask]

    return cache["ball"], cache["goal"]


def noisy_target_point_pos(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    ball_noise_std: tuple[float, float, float] = (0.03, 0.03, 0.02),
    goal_noise_std: tuple[float, float, float] = (0.03, 0.03, 0.02),
    noise_type: str = "normal",
    update_interval: int = 2,
    dropout_prob: float = 0.10,
    hold_last: bool = True,
) -> torch.Tensor:
    ball, _ = _noisy_soccer_target_pair(
        env,
        command_name,
        ball_noise_std=ball_noise_std,
        goal_noise_std=goal_noise_std,
        noise_type=noise_type,
        update_interval=update_interval,
        dropout_prob=dropout_prob,
        hold_last=hold_last,
    )
    return ball


def noisy_target_destination_pos_local(
    env: ManagerBasedEnv,
    command_name: str = "motion",
    ball_noise_std: tuple[float, float, float] = (0.03, 0.03, 0.02),
    goal_noise_std: tuple[float, float, float] = (0.03, 0.03, 0.02),
    noise_type: str = "normal",
    update_interval: int = 2,
    dropout_prob: float = 0.10,
    hold_last: bool = True,
) -> torch.Tensor:
    _, goal = _noisy_soccer_target_pair(
        env,
        command_name,
        ball_noise_std=ball_noise_std,
        goal_noise_std=goal_noise_std,
        noise_type=noise_type,
        update_interval=update_interval,
        dropout_prob=dropout_prob,
        hold_last=hold_last,
    )
    return goal


def blind_zone_target_point_pos(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    """Return target point in robot base frame with blind-zone simulation.
    
    If robot-ball (x, y) distance is outside [blind_distance_min, blind_distance_max],
    return the last visible position to emulate limited visibility.
    Thresholds are resampled from MotionCommandCfg ranges at each resample.
    """
    command = _get_motion_command(env, command_name)
    
    # Current target in robot base frame.
    target_base = get_target_point_base(env, command_name)
    
    # Compute robot-target (x, y) distance in world coordinates.
    target_world = get_target_point_world(env, command_name)
    robot_pos = command.robot_soccer_obs_pos_w
    # Horizontal distance only.
    distance_xy = torch.norm(target_world[:, :2] - robot_pos[:, :2], dim=-1)
    
    # Visible only when distance is within [min, max].
    in_visible_range = (distance_xy >= command.blind_distance_min) & (distance_xy <= command.blind_distance_max)
    
    # Update last visible target for visible environments.
    if torch.any(in_visible_range):
        command.last_visible_target_point_base[in_visible_range] = target_base[in_visible_range]
        command.is_in_blind_zone[in_visible_range] = False
    
    # Mark blind-zone environments.
    command.is_in_blind_zone[~in_visible_range] = True
    
    # Return last visible position in blind zone, otherwise current target.
    result = torch.where(
        command.is_in_blind_zone.unsqueeze(-1),
        command.last_visible_target_point_base,
        target_base
    )
    # print("blind zone target point:", command.blind_distance_min, command.blind_distance_max, result)
    return result


def target_destination_pos_local(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    if not hasattr(command, "target_destination_pos"):
        raise RuntimeError(f"motion command '{command_name}' lacks target_destination_pos attribute")
    # target_destination_pos is local to env origin; convert to world before subtracting robot pose.
    env_origins = getattr(env.scene, "env_origins", None)
    if env_origins is not None:
        target_world = command.target_destination_pos + env_origins
    else:
        target_world = command.target_destination_pos

    delta = target_world - command.robot_soccer_obs_pos_w
    # print("position:", quat_apply(quat_inv(command.robot_soccer_obs_quat_w), delta))
    return quat_apply(quat_inv(command.robot_soccer_obs_quat_w), delta)


def target_destination_pos_local_first_frame(env: ManagerBasedEnv, command_name: str = "motion") -> torch.Tensor:
    cache_name = f"_{command_name}_target_destination_local_cache"
    target_local = target_destination_pos_local(env, command_name)

    cache = getattr(env, cache_name, None)
    if cache is None or cache.shape[0] != env.num_envs:
        cache = target_local.clone()
        setattr(env, cache_name, cache)

    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is None:
        return cache

    first_step_mask = (step_buf == 0)
    if torch.any(first_step_mask):
        cache = getattr(env, cache_name)
        # Only refresh the cache when an environment just reset so the policy keeps the first-frame cue.
        cache[first_step_mask] = target_local[first_step_mask]
        setattr(env, cache_name, cache)
    # print("cache:", getattr(env, cache_name))
    return getattr(env, cache_name)
    # Positional encoding path is intentionally disabled here.
    return _positional_encoding(getattr(env, cache_name), num_freqs=6)


def target_destination_pos_local_ball_anchor_first_frame(
    env: ManagerBasedEnv,
    command_name: str = "motion",
) -> torch.Tensor:
    """Deployment-aligned student goal cue.

    The teacher observes the training target destination sampled by the environment. The student observes the same
    target intention as a first-frame ball-to-goal anchor, matching the deployment path where only an initial local
    ball cue and an intended ball-to-goal vector are available.
    """
    command = _get_motion_command(env, command_name)
    ball_local = target_point_pos_first_frame(env, command_name)

    current_yaw = yaw_quat(command.robot_soccer_obs_quat_w)
    cache_name = f"_{command_name}_start_pelvis_yaw_quat_cache"
    start_yaw = getattr(env, cache_name, None)
    if start_yaw is None or start_yaw.shape[0] != env.num_envs:
        start_yaw = current_yaw.clone()
        setattr(env, cache_name, start_yaw)

    step_buf = getattr(env, "episode_length_buf", None)
    if step_buf is None:
        goal_local = target_destination_pos_local(env, command_name)
        anchor_cache_name = f"_{command_name}_ball_to_goal_anchor_cache"
        if getattr(env, anchor_cache_name, None) is None:
            setattr(env, anchor_cache_name, goal_local - ball_local)
        return goal_local
    first_step_mask = step_buf == 0

    anchor_cache_name = f"_{command_name}_ball_to_goal_anchor_cache"
    anchor_cache = getattr(env, anchor_cache_name, None)
    if anchor_cache is None or anchor_cache.shape[0] != env.num_envs:
        goal_local = target_destination_pos_local(env, command_name)
        anchor_cache = goal_local - ball_local
        setattr(env, anchor_cache_name, anchor_cache)

    if torch.any(first_step_mask):
        start_yaw = getattr(env, cache_name)
        start_yaw[first_step_mask] = current_yaw[first_step_mask]
        setattr(env, cache_name, start_yaw)
        goal_local = target_destination_pos_local(env, command_name)
        anchor_cache = getattr(env, anchor_cache_name)
        anchor_cache[first_step_mask] = goal_local[first_step_mask] - ball_local[first_step_mask]
        setattr(env, anchor_cache_name, anchor_cache)

    relative_heading = quat_mul(quat_inv(current_yaw), getattr(env, cache_name))
    anchor_local = quat_apply(relative_heading, getattr(env, anchor_cache_name))
    return ball_local + anchor_local
    


def foot_target_point_distance(env: ManagerBasedEnv, robot_cfg: SceneEntityCfg, command_name: str = "motion",) -> torch.Tensor:
    command = _get_motion_command(env, command_name)
    robot = env.scene[robot_cfg.name]
    foot_pos = robot.data.body_pos_w[:, robot_cfg.body_ids]
    target_world = get_target_point_world(env, command_name)
    diff = foot_pos - target_world.unsqueeze(1)
    dist = torch.linalg.norm(diff, dim=-1)
    return dist.view(env.num_envs, -1)
