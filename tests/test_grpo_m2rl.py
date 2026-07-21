from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from grpo.data.m2rl import m2rl_frame_to_verl, validate_m2rl_frame
from grpo.rewards.m2rl import compute_gpqa_reward, compute_score
from mopd_verl.config_validation import validate_mopd_config
from mopd_verl.launch import build_overrides
from mopd_verl.paper_eval import run_paper_eval_from_config
from mopd_verl.prepare_data import VALID_TEACHERS, _load_optional_module, parse_args
from mopd_verl.settings import ModelConfig, load_config
from scripts.prepare_nemotron_rl_data import _science_row


class M2RLRewardTests(unittest.TestCase):
    def test_gpqa_reward_extracts_final_letter_without_think_tags(self) -> None:
        metadata = {"choices": ["alpha", "beta", "gamma", "delta"], "correct_letter": "C"}

        self.assertEqual(compute_gpqa_reward("The answer is C.", "C", metadata), 1.0)
        self.assertEqual(compute_gpqa_reward("Final answer: B", "C", metadata), 0.0)

    def test_gpqa_reward_uses_last_explicit_answer(self) -> None:
        metadata = {"choices": ["alpha", "beta", "gamma", "delta"], "correct_letter": "C"}

        response = "The answer is B, but after checking, final answer is C."
        self.assertEqual(compute_gpqa_reward(response, "C", metadata), 1.0)

    def test_gpqa_reward_requires_single_letter_answer(self) -> None:
        metadata = {"choices": ["alpha", "beta", "gamma", "delta"], "correct_letter": "C"}

        self.assertEqual(compute_gpqa_reward("The answer is CAT.", "C", metadata), 0.0)
        self.assertEqual(compute_gpqa_reward(r"Answer: \boxed{C}", "C", metadata), 1.0)
        self.assertEqual(compute_gpqa_reward("Answer: C_option", "C", metadata), 0.0)

    def test_gpqa_reward_ignores_placeholder_lists(self) -> None:
        metadata = {"choices": ["alpha", "beta", "gamma", "delta"], "correct_letter": "C"}

        response = r"Use Answer: \boxed{A/B/C/D}; after checking, I select <C>."
        self.assertEqual(compute_gpqa_reward(response, "C", metadata), 1.0)

        ten_choice_metadata = {
            "choices": [str(index) for index in range(10)],
            "correct_letter": "C",
            "valid_letters": list("ABCDEFGHIJ"),
        }
        response = r"Use Answer: \boxed{A/B/C/D/E/F/G/H/I/J}; answer B; then <C>."
        self.assertEqual(compute_gpqa_reward(response, "C", ten_choice_metadata), 1.0)

    def test_gpqa_reward_uses_dataset_output_regex(self) -> None:
        metadata = {
            "choices": ["alpha", "beta", "gamma", "delta"],
            "correct_letter": "C",
            "template_metadata": {"output_regex": r"<\s*([A-Za-z0-9])\s*>",},
        }

        response = "I first thought the answer is B; after checking I conclude <C>."
        self.assertEqual(compute_gpqa_reward(response, "C", metadata), 1.0)
        self.assertEqual(compute_gpqa_reward("The answer is CAT", "C", metadata), 0.0)

    def test_gpqa_reward_accepts_parquet_ndarray_metadata(self) -> None:
        metadata = {
            "choices": np.array(["alpha", "beta", "gamma", "delta"], dtype=object),
            "correct_letter": "C",
            "valid_letters": np.array(["A", "B", "C", "D"], dtype=object),
        }

        self.assertEqual(compute_gpqa_reward("Final answer: C", "C", metadata), 1.0)

    def test_gpqa_reward_normalizes_valid_letter_edge_cases(self) -> None:
        choices = ["alpha", "beta", "gamma", "delta"]
        for raw_letters in ["ABCD", "A,B,C,D", '["A", "B", "C", "D"]']:
            metadata = {"choices": choices, "correct_letter": "C", "valid_letters": raw_letters}
            self.assertEqual(compute_gpqa_reward("Final answer: C", "C", metadata), 1.0)

        for raw_letters in ["null", "42"]:
            metadata = {"choices": choices, "correct_letter": "C", "valid_letters": raw_letters}
            self.assertEqual(compute_gpqa_reward("Final answer: C", "C", metadata), 1.0)

        scalar_metadata = {
            "choices": choices,
            "correct_letter": "A",
            "valid_letters": np.array("A", dtype=object),
        }
        self.assertEqual(compute_gpqa_reward("Final answer: A", "A", scalar_metadata), 1.0)

    def test_compute_score_dispatches_science_from_data_source(self) -> None:
        score = compute_score(
            data_source="m2rl_science",
            solution_str="Option B is correct.",
            ground_truth="B",
            extra_info={"choices": ["A text", "B text"], "rm_type": "gpqa"},
        )

        self.assertEqual(score["score"], 1.0)
        self.assertEqual(score["m2rl_gpqa"], 1.0)


class M2RLDataTests(unittest.TestCase):
    def test_science_frame_converts_to_verl_schema(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "prompt": "Which option is right?",
                    "label": "B",
                    "metadata": {"choices": ["wrong", "right"], "correct_letter": "B"},
                }
            ]
        )

        report = validate_m2rl_frame(frame, rm_type="gpqa")
        self.assertTrue(report.is_valid)

        output = m2rl_frame_to_verl(frame, rm_type="gpqa", split="train", domain="science")
        row = output.iloc[0]
        self.assertEqual(row["data_source"], "m2rl_gpqa")
        self.assertEqual(row["prompt"][0]["role"], "user")
        self.assertEqual(row["reward_model"]["ground_truth"], "B")
        self.assertEqual(row["extra_info"]["rm_type"], "gpqa")
        self.assertEqual(row["extra_info"]["opd_teacher"], "science")

    def test_ifbench_validation_rejects_missing_instruction_metadata(self) -> None:
        frame = pd.DataFrame([{"prompt": "Obey two constraints.", "label": ""}])

        report = validate_m2rl_frame(frame, rm_type="ifbench")

        self.assertFalse(report.is_valid)
        self.assertIn("missing IFBench instruction_id_list metadata", report.invalid_rows[0]["reasons"])

    def test_ifbench_frame_with_metadata_converts(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "prompt": "Write exactly two words.",
                    "label": "",
                    "metadata": {
                        "instruction_id_list": ["count:word_count_range"],
                        "kwargs": [{"min_words": 2, "max_words": 2}],
                        "prompt_text": "Write exactly two words.",
                    },
                }
            ]
        )

        report = validate_m2rl_frame(frame, rm_type="ifbench")
        self.assertTrue(report.is_valid)
        normalized = m2rl_frame_to_verl(frame, rm_type="ifbench", split="train", domain="if")
        self.assertEqual(normalized.iloc[0]["extra_info"]["rm_type"], "ifbench")
        self.assertEqual(normalized.iloc[0]["extra_info"]["opd_teacher"], "if")

    def test_science_validation_rejects_answer_outside_valid_letters(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "prompt": "Which option is right?",
                    "label": "C",
                    "metadata": {
                        "choices": ["only option G"],
                        "valid_letters": ["G"],
                        "correct_letter": "C",
                    },
                }
            ]
        )

        report = validate_m2rl_frame(frame, rm_type="gpqa")

        self.assertFalse(report.is_valid)
        self.assertIn("not included in valid_letters", report.invalid_rows[0]["reasons"][0])

    def test_nemotron_converter_rejects_unscorable_science_row(self) -> None:
        row = {
            "category": "nano_v3_sft_profiled_stem_mcqa",
            "prompt": "Choose A or B.",
            "options": [{"A": "first"}, {"B": "second"}],
            "expected_answer": "C",
        }

        with self.assertRaisesRegex(ValueError, "not present in option labels"):
            _science_row(row, 0)

    def test_science_validation_derives_letters_from_choice_count(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "prompt": "Which option is right?",
                    "label": "C",
                    "metadata": {
                        "choices": ["first", "second"],
                        "correct_letter": "C",
                    },
                }
            ]
        )

        report = validate_m2rl_frame(frame, rm_type="gpqa")

        self.assertFalse(report.is_valid)
        self.assertIn("not included in valid_letters", report.invalid_rows[0]["reasons"][0])

    def test_science_validation_parses_json_encoded_choices(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "prompt": "Which option is right?",
                    "label": "Z",
                    "metadata": {
                        "choices": '["first", "second"]',
                        "correct_letter": "Z",
                    },
                }
            ]
        )

        report = validate_m2rl_frame(frame, rm_type="gpqa")

        self.assertFalse(report.is_valid)
        self.assertIn("not included in valid_letters", report.invalid_rows[0]["reasons"][0])


class PaperEvalCompatibilityTests(unittest.TestCase):
    def test_disabled_paper_eval_is_a_noop(self) -> None:
        self.assertEqual(run_paper_eval_from_config({}), {})
        self.assertEqual(run_paper_eval_from_config({"paper_eval": {"enabled": False}}), {})

    def test_enabled_paper_eval_fails_with_actionable_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "paper_eval is not available"):
            run_paper_eval_from_config({"paper_eval": {"enabled": True}})


class ScienceConfigTests(unittest.TestCase):
    def test_model_config_defaults_to_flash_attention_two(self) -> None:
        self.assertEqual(ModelConfig.attn_implementation, "flash_attention_2")
        self.assertFalse(ModelConfig.use_remove_padding)

    def test_actor_is_initialized_in_fp32_for_optimizer_correctness(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_config(project_root / "grpo/configs/m2rl_science.yaml")

        self.assertEqual(config.actor.model_dtype, "fp32")
        self.assertIn(
            "actor_rollout_ref.actor.fsdp_config.model_dtype=fp32",
            build_overrides(config),
        )

    def test_cross_field_validation_rejects_non_divisible_actor_batch(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_config(project_root / "grpo/configs/m2rl_science.yaml")
        invalid = replace(
            config,
            actor=replace(config.actor, ppo_mini_batch_size=31),
            trainer=replace(config.trainer, n_gpus_per_node=6),
        )

        with self.assertRaisesRegex(ValueError, "must be divisible"):
            validate_mopd_config(invalid)

    def test_cross_field_validation_checks_train_rollout_batch(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config = load_config(project_root / "grpo/configs/m2rl_science.yaml")
        invalid = replace(
            config,
            data=replace(config.data, train_batch_size=25),
            trainer=replace(config.trainer, n_gpus_per_node=6),
        )

        with self.assertRaisesRegex(ValueError, "data.train_batch_size"):
            validate_mopd_config(invalid)

    def test_m2rl_configs_force_safe_flash_attention_two_defaults(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        config_paths = (
            "grpo/configs/m2rl_if.yaml",
            "grpo/configs/m2rl_if_smoke.yaml",
            "grpo/configs/m2rl_science.yaml",
            "grpo/configs/m2rl_science_smoke_2gpu.yaml",
            "grpo/configs/m2rl_if_science_mix.yaml",
        )

        for relative_path in config_paths:
            with self.subTest(config=relative_path):
                config = load_config(project_root / relative_path)
                overrides = build_overrides(config)
                self.assertIn(
                    "+actor_rollout_ref.model.override_config.attn_implementation="
                    "flash_attention_2",
                    overrides,
                )
                self.assertIn(
                    "actor_rollout_ref.model.use_remove_padding=False",
                    overrides,
                )

    def test_mopd_verl_exposes_m2rl_data_preparation(self) -> None:
        args = parse_args(
            [
                "prepare-m2rl",
                "--input",
                "science.jsonl",
                "--output",
                "science.parquet",
                "--rm-type",
                "gpqa",
                "--domain",
                "science",
            ]
        )

        self.assertEqual(args.command, "prepare-m2rl")
        self.assertEqual(args.domain, "science")
        self.assertIn("if", VALID_TEACHERS)
        self.assertIn("science", VALID_TEACHERS)

    def test_optional_module_loader_preserves_transitive_import_error(self) -> None:
        transitive_error = ModuleNotFoundError(
            "No module named 'missing_dependency'",
            name="missing_dependency",
        )

        with patch(
            "mopd_verl.prepare_data.import_module",
            side_effect=transitive_error,
        ):
            with self.assertRaises(ModuleNotFoundError) as context:
                _load_optional_module("optional.module", "optional-command")

        self.assertIs(context.exception, transitive_error)


if __name__ == "__main__":
    unittest.main()
