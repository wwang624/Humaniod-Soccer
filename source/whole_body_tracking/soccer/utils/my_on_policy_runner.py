import os

import torch
from rsl_rl.algorithms import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.modules.student_teacher_recurrent import StudentTeacherRecurrent
from rsl_rl.networks import EmpiricalNormalization

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
        if rollout_mode not in {"mean", "sample"}:
            raise ValueError(f"Unsupported SOCCER_DISTILL_ROLLOUT_MODE={rollout_mode!r}; expected 'mean' or 'sample'.")
        self.freeze_student_obs_normalizer = freeze_student_obs_normalizer
        self.rollout_mode = rollout_mode
        if self.freeze_student_obs_normalizer:
            print("[INFO] MotionDistillation: freezing student observation normalizer updates.")
        print(f"[INFO] MotionDistillation: rollout_mode={self.rollout_mode}.")

    def act(self, obs):
        if self.rollout_mode == "sample":
            self.transition.actions = self.policy.act(obs).detach()
        else:
            self.transition.actions = self.policy.act_inference(obs).detach()
        self.transition.privileged_actions = self.policy.evaluate(obs).detach()
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
