from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mopd_verl.checkpoints import checkpoint_retention_limit, prune_global_checkpoints


class CheckpointRetentionTests(unittest.TestCase):
    def test_pruning_scans_existing_checkpoints_after_restart(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for step in (10, 20, 30, 40, 50, 60, 70):
                checkpoint = root / f"global_step_{step}"
                (checkpoint / "actor").mkdir(parents=True)
                (checkpoint / "data.pt").write_text("state", encoding="utf-8")
            (root / "unrelated").mkdir()

            removed = prune_global_checkpoints(root, 5)

            self.assertEqual([path.name for path in removed], ["global_step_10", "global_step_20"])
            self.assertEqual(
                sorted(path.name for path in root.glob("global_step_*")),
                [
                    "global_step_30",
                    "global_step_40",
                    "global_step_50",
                    "global_step_60",
                    "global_step_70",
                ],
            )
            self.assertTrue((root / "unrelated").is_dir())

    def test_pruning_does_not_count_partial_checkpoint_as_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for step in (10, 20, 30, 40, 50, 70):
                checkpoint = root / f"global_step_{step}"
                (checkpoint / "actor").mkdir(parents=True)
                (checkpoint / "data.pt").write_text("state", encoding="utf-8")
            (root / "global_step_60").mkdir()

            prune_global_checkpoints(root, 5)

            self.assertEqual(
                sorted(path.name for path in root.glob("global_step_*")),
                [
                    "global_step_20",
                    "global_step_30",
                    "global_step_40",
                    "global_step_50",
                    "global_step_70",
                ],
            )

    def test_complete_checkpoint_limit_uses_stricter_component(self) -> None:
        self.assertEqual(checkpoint_retention_limit(5, 3, use_critic=True), 3)
        self.assertEqual(checkpoint_retention_limit(5, None, use_critic=False), 5)
        self.assertIsNone(checkpoint_retention_limit(None, None, use_critic=False))


if __name__ == "__main__":
    unittest.main()
