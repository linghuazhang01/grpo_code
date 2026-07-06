from __future__ import annotations

import unittest

import pandas as pd

from grpo.data.m2rl import m2rl_frame_to_verl, validate_m2rl_frame
from grpo.rewards.m2rl import compute_gpqa_reward, compute_score


class M2RLRewardTests(unittest.TestCase):
    def test_gpqa_reward_extracts_final_letter_without_think_tags(self) -> None:
        metadata = {"choices": ["alpha", "beta", "gamma", "delta"], "correct_letter": "C"}

        self.assertEqual(compute_gpqa_reward("The answer is C.", "C", metadata), 1.0)
        self.assertEqual(compute_gpqa_reward("Final answer: B", "C", metadata), 0.0)

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


if __name__ == "__main__":
    unittest.main()
