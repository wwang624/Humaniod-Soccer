import os

import torch
import torch.nn as nn
from rsl_rl.algorithms import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.networks import MLP, Memory
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.modules.student_teacher_recurrent import StudentTeacherRecurrent
from rsl_rl.networks import EmpiricalNormalization
from torch.distributions import Normal

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from soccer.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MotionStudentTeacherRecurrent(StudentTeacherRecurrent):
    """Recurrent student-teacher module with PPO recurrent-teacher checkpoint loading."""

    def __init__(self, *args, teacher_uses_student_obs: bool | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if teacher_uses_student_obs is None:
            value = os.environ.get("SOCCER_TEACHER_USES_STUDENT_OBS", "0").strip().lower()
            teacher_uses_student_obs = value in {"1", "true", "yes", "on"}
        self.teacher_uses_student_obs = teacher_uses_student_obs
        if self.teacher_uses_student_obs:
            print("[INFO] MotionStudentTeacherRecurrent: teacher labels use the student policy observation tensor.")
        if self.teacher_recurrent:
            self.teacher_input_obs_normalizer = EmpiricalNormalization(self.memory_t.rnn.input_size)
        else:
            self.teacher_input_obs_normalizer = torch.nn.Identity()

    @staticmethod
    def _can_load_state_dict(module, state_dict) -> bool:
        module_state_dict = module.state_dict()
        if module_state_dict.keys() != state_dict.keys():
            return False
        return all(module_state_dict[key].shape == value.shape for key, value in state_dict.items())

    @property
    def action_std(self):
        if getattr(self, "distribution", None) is not None:
            return self.distribution.stddev
        if self.noise_std_type == "scalar":
            return self.std
        return self.log_std.exp()

    def load_state_dict(self, state_dict, strict=True):
        if any("actor" in key for key in state_dict.keys()):
            teacher_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("actor."):
                    teacher_state_dict[key.replace("actor.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            initialized_student_actor = self._can_load_state_dict(self.student, teacher_state_dict)
            if initialized_student_actor:
                self.student.load_state_dict(teacher_state_dict, strict=strict)

            actor_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("actor_obs_normalizer."):
                    actor_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            loaded_teacher_normalizer = self._can_load_state_dict(
                self.teacher_input_obs_normalizer, actor_obs_normalizer_state_dict
            )
            if loaded_teacher_normalizer:
                self.teacher_input_obs_normalizer.load_state_dict(actor_obs_normalizer_state_dict, strict=strict)
            loaded_student_normalizer = self._can_load_state_dict(
                self.student_obs_normalizer, actor_obs_normalizer_state_dict
            )
            if loaded_student_normalizer:
                self.student_obs_normalizer.load_state_dict(actor_obs_normalizer_state_dict, strict=strict)

            if self.teacher_recurrent:
                teacher_memory_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("memory_a."):
                        teacher_memory_state_dict[key.replace("memory_a.", "")] = value
                if not teacher_memory_state_dict:
                    raise RuntimeError("Recurrent teacher checkpoint is missing memory_a.* weights.")
                self.memory_t.load_state_dict(teacher_memory_state_dict, strict=strict)
                initialized_student_memory = self._can_load_state_dict(self.memory_s, teacher_memory_state_dict)
                if initialized_student_memory:
                    self.memory_s.load_state_dict(teacher_memory_state_dict, strict=strict)
                self.memory_t.eval()
            else:
                teacher_memory_state_dict = {}
                initialized_student_memory = False

            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_input_obs_normalizer.eval()
            print(
                "[INFO] Loaded recurrent distillation teacher: "
                f"actor_keys={len(teacher_state_dict)} memory_keys={len(teacher_memory_state_dict)}. "
                f"student_actor_init={initialized_student_actor} student_memory_init={initialized_student_memory} "
                f"teacher_normalizer={loaded_teacher_normalizer} student_normalizer={loaded_student_normalizer}"
            )
            return False

        return super().load_state_dict(state_dict, strict=strict)

    def evaluate(self, obs):
        obs = self.get_teacher_obs(obs)
        obs = self.teacher_input_obs_normalizer(obs)
        with torch.no_grad():
            if self.teacher_recurrent:
                self.memory_t.eval()
                obs = self.memory_t(obs).squeeze(0)
            return self.teacher(obs)

    def get_teacher_obs(self, obs):
        if self.teacher_uses_student_obs:
            return self.get_student_obs(obs)
        return super().get_teacher_obs(obs)

    def train(self, mode=True):
        super().train(mode)
        self.teacher_input_obs_normalizer.eval()


class MotionStudentTeacherMLPPhase(nn.Module):
    """MLP student distilled from a recurrent PPO teacher without student warm-start."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[128, 64, 32],
        activation="elu",
        init_noise_std=0.1,
        noise_std_type: str = "scalar",
        rnn_type="lstm",
        rnn_hidden_dim=128,
        rnn_num_layers=2,
        **kwargs,
    ):
        if kwargs:
            print(
                "MotionStudentTeacherMLPPhase.__init__ got unexpected arguments, ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()
        self.loaded_teacher = False
        self.obs_groups = obs_groups

        num_student_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "MLP-history student only supports 1D observations."
            num_student_obs += obs[obs_group].shape[-1]
        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, "Teacher only supports 1D observations."
            num_teacher_obs += obs[obs_group].shape[-1]

        self.student = MLP(num_student_obs, num_actions, student_hidden_dims, activation)
        self.student_obs_normalization = student_obs_normalization
        self.student_obs_normalizer = EmpiricalNormalization(num_student_obs) if student_obs_normalization else nn.Identity()

        self.memory_t = Memory(num_teacher_obs, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_dim)
        self.teacher = MLP(rnn_hidden_dim, num_actions, teacher_hidden_dims, activation)
        self.teacher_obs_normalization = teacher_obs_normalization
        self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs) if teacher_obs_normalization else nn.Identity()
        self.teacher_input_obs_normalizer = EmpiricalNormalization(num_teacher_obs)

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")
        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"MLP phase student input dim: {num_student_obs}")
        print(f"MLP phase student: {self.student}")
        print(f"Recurrent teacher input dim: {num_teacher_obs}")
        print(f"Recurrent teacher RNN: {self.memory_t}")
        print(f"Recurrent teacher MLP: {self.teacher}")

    @property
    def action_std(self):
        if getattr(self, "distribution", None) is not None:
            return self.distribution.stddev
        if self.noise_std_type == "scalar":
            return self.std
        return self.log_std.exp()

    def get_student_obs(self, obs):
        return torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)

    def get_teacher_obs(self, obs):
        return torch.cat([obs[group] for group in self.obs_groups["teacher"]], dim=-1)

    def update_distribution(self, obs):
        mean = self.student(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs):
        obs = self.student_obs_normalizer(self.get_student_obs(obs))
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.student_obs_normalizer(self.get_student_obs(obs))
        return self.student(obs)

    def evaluate(self, obs):
        obs = self.teacher_input_obs_normalizer(self.get_teacher_obs(obs))
        with torch.no_grad():
            self.memory_t.eval()
            out = self.memory_t(obs).squeeze(0)
            return self.teacher(out)

    def reset(self, dones=None, hidden_states=None):
        teacher_hidden = None
        if hidden_states is not None:
            teacher_hidden = hidden_states[1] if isinstance(hidden_states, tuple) else hidden_states
        self.memory_t.reset(dones, teacher_hidden)

    def get_hidden_states(self):
        return None, self.memory_t.hidden_states

    def detach_hidden_states(self, dones=None):
        self.memory_t.detach_hidden_states(dones)

    def update_normalization(self, obs):
        if self.student_obs_normalization:
            self.student_obs_normalizer.update(self.get_student_obs(obs))

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        self.memory_t.eval()
        self.teacher_obs_normalizer.eval()
        self.teacher_input_obs_normalizer.eval()

    @staticmethod
    def _can_load_state_dict(module, state_dict) -> bool:
        module_state_dict = module.state_dict()
        if module_state_dict.keys() != state_dict.keys():
            return False
        return all(module_state_dict[key].shape == value.shape for key, value in state_dict.items())

    def load_state_dict(self, state_dict, strict=True):
        if any(key.startswith("actor.") for key in state_dict.keys()):
            teacher_state_dict = {
                key.replace("actor.", ""): value for key, value in state_dict.items() if key.startswith("actor.")
            }
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)

            actor_obs_normalizer_state_dict = {
                key.replace("actor_obs_normalizer.", ""): value
                for key, value in state_dict.items()
                if key.startswith("actor_obs_normalizer.")
            }
            loaded_teacher_normalizer = self._can_load_state_dict(
                self.teacher_input_obs_normalizer, actor_obs_normalizer_state_dict
            )
            if loaded_teacher_normalizer:
                self.teacher_input_obs_normalizer.load_state_dict(actor_obs_normalizer_state_dict, strict=strict)

            teacher_memory_state_dict = {
                key.replace("memory_a.", ""): value for key, value in state_dict.items() if key.startswith("memory_a.")
            }
            if not teacher_memory_state_dict:
                raise RuntimeError("Recurrent teacher checkpoint is missing memory_a.* weights.")
            self.memory_t.load_state_dict(teacher_memory_state_dict, strict=strict)

            self.loaded_teacher = True
            self.teacher.eval()
            self.memory_t.eval()
            self.teacher_input_obs_normalizer.eval()
            print(
                "[INFO] Loaded recurrent teacher for MLP phase distillation: "
                f"actor_keys={len(teacher_state_dict)} memory_keys={len(teacher_memory_state_dict)} "
                f"teacher_normalizer={loaded_teacher_normalizer} student_init_from_teacher=False"
            )
            return False
        if any(key.startswith("student.") for key in state_dict.keys()):
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.memory_t.eval()
            self.teacher_input_obs_normalizer.eval()
            return True
        raise ValueError("state_dict does not contain teacher actor.* or distillation student.* parameters")


MotionStudentTeacherMLPHistory = MotionStudentTeacherMLPPhase


class MotionDistillation(Distillation):
    """Distillation with configurable student rollout actions."""

    def __init__(
        self,
        *args,
        freeze_student_obs_normalizer: bool | None = None,
        rollout_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if freeze_student_obs_normalizer is None:
            value = os.environ.get("SOCCER_FREEZE_STUDENT_OBS_NORMALIZER", "0").strip().lower()
            freeze_student_obs_normalizer = value in {"1", "true", "yes", "on"}
        if rollout_mode is None:
            rollout_mode = os.environ.get("SOCCER_DISTILL_ROLLOUT_MODE", "mean").strip().lower()
        if rollout_mode not in {"mean", "sample", "teacher", "mix"}:
            raise ValueError(
                f"Unsupported SOCCER_DISTILL_ROLLOUT_MODE={rollout_mode!r}; "
                "expected 'mean', 'sample', 'teacher', or 'mix'."
            )
        self.freeze_student_obs_normalizer = freeze_student_obs_normalizer
        self.rollout_mode = rollout_mode
        self.teacher_mix_alpha = float(os.environ.get("SOCCER_DISTILL_TEACHER_MIX_ALPHA", "1.0"))
        if self.freeze_student_obs_normalizer:
            print("[INFO] MotionDistillation: freezing student observation normalizer updates.")
        print(
            f"[INFO] MotionDistillation: rollout_mode={self.rollout_mode} "
            f"teacher_mix_alpha={self.teacher_mix_alpha:.3f}."
        )

    @staticmethod
    def _clone_hidden_states(hidden_states):
        if hidden_states is None:
            return None

        def clone_one(value):
            if value is None:
                return None
            if isinstance(value, tuple):
                return tuple(clone_one(item) for item in value)
            return value.detach().clone()

        return tuple(clone_one(value) for value in hidden_states)

    def act(self, obs):
        if getattr(self.policy, "is_recurrent", False):
            self.transition.hidden_states = self.policy.get_hidden_states()
        teacher_actions = self.policy.evaluate(obs).detach()
        if self.rollout_mode == "sample":
            self.transition.actions = self.policy.act(obs).detach()
        elif self.rollout_mode == "teacher":
            self.transition.actions = teacher_actions
        elif self.rollout_mode == "mix":
            alpha = min(max(self.teacher_mix_alpha, 0.0), 1.0)
            student_mean_actions = self.policy.act_inference(obs).detach()
            self.transition.actions = alpha * teacher_actions + (1.0 - alpha) * student_mean_actions
        else:
            self.transition.actions = self.policy.act_inference(obs).detach()
        self.transition.privileged_actions = teacher_actions
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        if not self.freeze_student_obs_normalizer:
            self.policy.update_normalization(obs)

        self.transition.rewards = rewards
        self.transition.dones = dones
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self):
        rollout_start_hidden_states = self._clone_hidden_states(self.last_hidden_states)
        rollout_end_hidden_states = self._clone_hidden_states(self.policy.get_hidden_states())

        self.num_updates += 1
        mean_behavior_loss = 0.0
        mean_waist_behavior_loss = 0.0
        loss = 0
        cnt = 0

        waist_action_indices = [2, 5, 8]

        for _ in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self._clone_hidden_states(rollout_start_hidden_states))
            self.policy.detach_hidden_states()
            for obs, _, privileged_actions, dones in self.storage.generator():
                actions = self.policy.act_inference(obs)

                behavior_loss = self.loss_fn(actions, privileged_actions)
                action_sq_error = (actions - privileged_actions).pow(2)
                valid_waist_indices = [idx for idx in waist_action_indices if idx < action_sq_error.shape[-1]]
                if valid_waist_indices:
                    waist_behavior_loss = action_sq_error[..., valid_waist_indices].mean()
                    mean_waist_behavior_loss += waist_behavior_loss.item()

                loss = loss + behavior_loss
                mean_behavior_loss += behavior_loss.item()
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

            if loss != 0:
                self.optimizer.zero_grad()
                loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                if self.max_grad_norm:
                    nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.policy.detach_hidden_states()
                loss = 0

        if cnt == 0:
            raise RuntimeError("MotionDistillation.update() called with empty rollout storage.")

        mean_behavior_loss /= cnt
        mean_waist_behavior_loss = mean_waist_behavior_loss / cnt if mean_waist_behavior_loss else 0.0
        self.storage.clear()
        self.last_hidden_states = rollout_end_hidden_states
        self.policy.reset(hidden_states=self._clone_hidden_states(rollout_end_hidden_states))
        self.policy.detach_hidden_states()

        return {
            "behavior": mean_behavior_loss,
            "behavior/waist": mean_waist_behavior_loss,
        }


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
