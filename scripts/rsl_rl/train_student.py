# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--registry_name", type=str, required=False, help="The name of the wand registry.")
parser.add_argument("--motion_path", type=str, required=True, help="The path to the motion file or directory containing motion files.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import glob
import pickle
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import rsl_rl.runners.distillation_runner as rsl_distillation_runner_module
import rsl_rl.runners.on_policy_runner as rsl_runner_module
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

# Import extensions to set up environment tasks
import soccer.tasks  # noqa: F401
from soccer.utils.my_on_policy_runner import MotionDistillation, MotionStudentTeacherRecurrent

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def dump_pickle(filename: str, data):
    """Compatibility helper for IsaacLab versions that no longer export dump_pickle."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def get_motion_files(motion_path: str) -> list[str]:
    """
    Get a list of motion files.
    
    Args:
        motion_path: File path or directory path.
        
    Returns:
        List of motion file paths.
    """
    if os.path.isfile(motion_path):
        # Single-file input.
        return [motion_path]
    elif os.path.isdir(motion_path):
        # Directory input: collect all .npz files.
        motion_files = glob.glob(os.path.join(motion_path, "*.npz"))
        if not motion_files:
            raise ValueError(f"No .npz files found in directory: {motion_path}")
        motion_files.sort()
        print(f"Found {len(motion_files)} motion files in {motion_path}")
        for file in motion_files:
            print(f"  - {os.path.basename(file)}")
        return motion_files
    else:
        raise ValueError(f"Invalid path: {motion_path}. Must be a file or directory.")


def resolve_checkpoint_path(log_root_path: str, load_run: str, load_checkpoint: str) -> str:
    if load_run.endswith(".pt"):
        if os.path.isabs(load_run):
            return load_run
        return os.path.join(log_root_path, load_run)
    if os.path.isabs(load_run) or os.path.isdir(load_run) or os.sep in load_run:
        run_path = load_run if os.path.isabs(load_run) else os.path.abspath(load_run)
        if not os.path.isdir(run_path):
            run_path = os.path.join(log_root_path, load_run)
        checkpoint_path = os.path.join(run_path, load_checkpoint)
        if os.path.isfile(checkpoint_path):
            return checkpoint_path
        return get_checkpoint_path(os.path.dirname(run_path), os.path.basename(run_path), load_checkpoint)
    return get_checkpoint_path(log_root_path, load_run, load_checkpoint)


def is_distillation_run(agent_cfg: RslRlBaseRunnerCfg) -> bool:
    return getattr(agent_cfg, "class_name", None) == "DistillationRunner" or getattr(
        agent_cfg.algorithm, "class_name", None
    ) in {"Distillation", "MotionDistillation"}


def get_runner_class(class_name: str):
    if class_name == "DistillationRunner":
        return DistillationRunner
    if class_name == "OnPolicyRunner":
        return OnPolicyRunner
    raise ValueError(f"Unsupported RSL-RL runner class: {class_name}")


def load_runner_checkpoint(runner, path: str, *, distillation_run: bool):
    if not distillation_run:
        return runner.load(path)

    loaded_dict = torch.load(path, weights_only=False)
    model_state_dict = dict(loaded_dict["model_state_dict"])
    loading_teacher_checkpoint = any(key.startswith("actor.") for key in model_state_dict)
    if loading_teacher_checkpoint and "obs_norm_state_dict" in loaded_dict:
        for key, value in loaded_dict["obs_norm_state_dict"].items():
            model_state_dict[f"actor_obs_normalizer.{key}"] = value

    resumed_training = runner.alg.policy.load_state_dict(model_state_dict)
    if resumed_training:
        runner.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        runner.current_learning_iteration = loaded_dict["iter"]
    return loaded_dict.get("infos")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    motion_files = get_motion_files(args_cli.motion_path)

    env_cfg.commands.motion.motion_files = motion_files

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or is_distillation_run(agent_cfg):
        resume_path = resolve_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    rsl_runner_module.MotionDistillation = MotionDistillation
    rsl_runner_module.MotionStudentTeacherRecurrent = MotionStudentTeacherRecurrent
    rsl_distillation_runner_module.MotionDistillation = MotionDistillation
    rsl_distillation_runner_module.MotionStudentTeacherRecurrent = MotionStudentTeacherRecurrent
    runner_class = get_runner_class(agent_cfg.class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or is_distillation_run(agent_cfg):
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        load_runner_checkpoint(runner, resume_path, distillation_run=is_distillation_run(agent_cfg))

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=not is_distillation_run(agent_cfg),
    )

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
