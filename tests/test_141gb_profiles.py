from __future__ import annotations

import os
import shlex
import subprocess
import unittest
from pathlib import Path

from mopd_verl.settings import load_config


class LargeGpuTrainingProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]

    def _dry_run(self, script_name: str) -> dict[str, str]:
        environment = os.environ.copy()
        environment.update(
            {
                "GRPO_MODEL_PATH": "/models/Qwen3-4B",
                "LC_ALL": "C",
                "LANG": "C",
            }
        )
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        result = subprocess.run(
            [str(self.project_root / "scripts" / script_name), "--dry-run"],
            cwd=self.project_root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        overrides: dict[str, str] = {}
        for token in shlex.split(result.stdout):
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            overrides[key.lstrip("+")] = value
        return overrides

    def test_science_and_if_profiles_have_consistent_141gb_limits(self) -> None:
        cases = {
            "run_m2rl_science_6gpu_141gb.sh": (6, 126, 2016, "science"),
            "run_m2rl_science_8gpu_141gb.sh": (8, 128, 2048, "science"),
            "run_m2rl_if_6gpu_141gb.sh": (6, 126, 2016, "if"),
            "run_m2rl_if_8gpu_141gb.sh": (8, 128, 2048, "if"),
        }

        for script_name, (
            gpu_count,
            train_batch_size,
            expected_trajectories,
            domain,
        ) in cases.items():
            with self.subTest(script=script_name):
                overrides = self._dry_run(script_name)
                self.assertEqual(overrides["trainer.n_gpus_per_node"], str(gpu_count))
                self.assertEqual(overrides["data.train_batch_size"], str(train_batch_size))
                self.assertEqual(overrides["data.max_response_length"], "16384")
                self.assertEqual(overrides["actor_rollout_ref.rollout.max_model_len"], "18432")
                self.assertEqual(overrides["actor_rollout_ref.rollout.n"], "16")
                self.assertEqual(overrides["actor_rollout_ref.rollout.tensor_model_parallel_size"], "1")
                self.assertEqual(overrides["actor_rollout_ref.actor.fsdp_config.model_dtype"], "fp32")
                self.assertEqual(overrides["actor_rollout_ref.actor.use_kl_loss"], "False")
                self.assertEqual(overrides["actor_rollout_ref.actor.kl_loss_coef"], "0.0")
                self.assertEqual(overrides["actor_rollout_ref.actor.entropy_coeff"], "0.0")
                self.assertEqual(overrides["algorithm.use_kl_in_reward"], "False")
                self.assertEqual(overrides["algorithm.kl_ctrl.kl_coef"], "0.0")
                self.assertEqual(overrides["actor_rollout_ref.ref.fsdp_config.param_offload"], "False")
                self.assertEqual(overrides["algorithm.rollout_correction.rollout_is_threshold"], "5.0")
                self.assertEqual(overrides["trainer.save_freq"], "10")
                self.assertEqual(overrides["trainer.max_actor_ckpt_to_keep"], "5")
                self.assertEqual(overrides["trainer.max_critic_ckpt_to_keep"], "5")
                self.assertEqual(overrides["trainer.total_training_steps"], "1200")
                self.assertEqual(overrides["trainer.test_freq"], "20")
                self.assertEqual(overrides["trainer.resume_mode"], "auto")
                self.assertIn(domain, overrides["trainer.experiment_name"])
                self.assertIn(f"{gpu_count}gpu", overrides["trainer.experiment_name"])

                rollout_batch_size = train_batch_size * int(
                    overrides["actor_rollout_ref.rollout.n"]
                )
                self.assertEqual(rollout_batch_size, expected_trajectories)
                self.assertEqual(rollout_batch_size % gpu_count, 0)

    def test_each_141gb_profile_has_a_standalone_config(self) -> None:
        cases = {
            "m2rl_science_6gpu_141gb.yaml": (6, 126, "science"),
            "m2rl_science_8gpu_141gb.yaml": (8, 128, "science"),
            "m2rl_if_6gpu_141gb.yaml": (6, 126, "if"),
            "m2rl_if_8gpu_141gb.yaml": (8, 128, "if"),
        }

        for filename, (gpu_count, prompt_batch, domain) in cases.items():
            with self.subTest(config=filename):
                config = load_config(self.project_root / "grpo/configs" / filename)
                self.assertEqual(config.trainer.n_gpus_per_node, gpu_count)
                self.assertEqual(config.data.train_batch_size, prompt_batch)
                self.assertEqual(config.actor.ppo_mini_batch_size, prompt_batch)
                self.assertEqual(config.rollout.n, 16)
                self.assertEqual(config.data.max_response_length, 16384)
                self.assertFalse(config.actor.use_kl_loss)
                self.assertEqual(config.actor.kl_loss_coef, 0.0)
                self.assertEqual(config.actor.entropy_coeff, 0.0)
                self.assertEqual(config.trainer.resume_mode, "auto")
                self.assertEqual(config.trainer.test_freq, 20)
                self.assertFalse(config.model.reference_param_offload)
                self.assertEqual(config.rollout_correction.rollout_is_threshold, 5.0)
                self.assertIn(domain, config.trainer.experiment_name)

    def test_two_gpu_science_smoke_profile_is_short_and_memory_bounded(self) -> None:
        overrides = self._dry_run("run_m2rl_science_2gpu_smoke.sh")

        self.assertEqual(overrides["trainer.n_gpus_per_node"], "2")
        self.assertEqual(overrides["data.train_batch_size"], "4")
        self.assertEqual(overrides["data.max_response_length"], "512")
        self.assertEqual(overrides["actor_rollout_ref.actor.ppo_mini_batch_size"], "4")
        self.assertEqual(overrides["actor_rollout_ref.actor.ppo_max_token_len_per_gpu"], "4096")
        self.assertEqual(overrides["actor_rollout_ref.actor.fsdp_config.model_dtype"], "fp32")
        self.assertEqual(overrides["actor_rollout_ref.rollout.n"], "4")
        self.assertEqual(overrides["actor_rollout_ref.rollout.gpu_memory_utilization"], "0.5")
        self.assertEqual(overrides["actor_rollout_ref.rollout.max_model_len"], "2560")
        self.assertEqual(overrides["actor_rollout_ref.rollout.max_num_batched_tokens"], "4096")
        self.assertEqual(overrides["actor_rollout_ref.rollout.max_num_seqs"], "8")
        self.assertEqual(overrides["trainer.total_training_steps"], "2")
        self.assertEqual(overrides["trainer.save_freq"], "-1")
        self.assertEqual(overrides["trainer.logger"], '["console"]')

        rollout_batch_size = int(overrides["data.train_batch_size"]) * int(
            overrides["actor_rollout_ref.rollout.n"]
        )
        self.assertEqual(rollout_batch_size, 16)
        self.assertEqual(rollout_batch_size % 2, 0)

    def test_two_gpu_smoke_profile_rejects_visible_gpu_count_mismatch(self) -> None:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "0"
        result = subprocess.run(
            [
                str(self.project_root / "scripts/run_m2rl_science_2gpu_smoke.sh"),
                "--dry-run",
            ],
            cwd=self.project_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("exactly 2 GPUs", result.stderr)

    def test_profile_rejects_visible_gpu_count_mismatch(self) -> None:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "0,1"
        result = subprocess.run(
            [
                str(self.project_root / "scripts/run_m2rl_science_6gpu_141gb.sh"),
                "--dry-run",
            ],
            cwd=self.project_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("exposes 2 GPUs", result.stderr)

    def test_profile_rejects_protected_hydra_override(self) -> None:
        result = subprocess.run(
            [
                str(self.project_root / "scripts/run_m2rl_science_6gpu_141gb.sh"),
                "--dry-run",
                "--",
                "trainer.total_training_steps=1",
            ],
            cwd=self.project_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Protected 141GB profile setting", result.stderr)

    def test_profile_rejects_parent_mapping_override(self) -> None:
        result = subprocess.run(
            [
                str(self.project_root / "scripts/run_m2rl_science_6gpu_141gb.sh"),
                "--dry-run",
                "--",
                "actor_rollout_ref.actor.fsdp_config={model_dtype: bfloat16}",
            ],
            cwd=self.project_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Protected 141GB profile setting", result.stderr)

    def test_profile_rejects_attention_and_thinking_overrides(self) -> None:
        protected_overrides = (
            "+data.apply_chat_template_kwargs.enable_thinking=True",
            "actor_rollout_ref.model.override_config.attn_implementation=eager",
            "actor_rollout_ref.model.use_remove_padding=True",
            "actor_rollout_ref.actor.use_kl_loss=True",
            "actor_rollout_ref.actor.kl_loss_coef=0.1",
            "actor_rollout_ref.actor.entropy_coeff=0.001",
            "algorithm.use_kl_in_reward=True",
            "algorithm.kl_ctrl.kl_coef=0.1",
            "algorithm={use_kl_in_reward:true,kl_ctrl:{type:fixed,kl_coef:0.1}}",
            "actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages=True",
            "actor_rollout_ref.actor.policy_loss.multi_teacher_distill=True",
            "actor_rollout_ref.actor.policy_loss.loss_mode=kl-cov",
            "actor_rollout_ref.actor.policy_loss={loss_mode:kl-cov}",
        )

        for override in protected_overrides:
            with self.subTest(override=override):
                result = subprocess.run(
                    [
                        str(self.project_root / "scripts/run_m2rl_science_8gpu_141gb.sh"),
                        "--dry-run",
                        "--",
                        override,
                    ],
                    cwd=self.project_root,
                    check=False,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(result.returncode, 2)
                self.assertIn("Protected 141GB profile setting", result.stderr)


if __name__ == "__main__":
    unittest.main()
