from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import mopd_verl.launch as launcher
from mopd_verl.settings import RuntimeConfig, load_config


class WandbEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_root = Path(__file__).resolve().parents[1]
        self.config = load_config(self.project_root / "grpo/configs/m2rl_science.yaml")

    def test_env_local_is_the_default(self) -> None:
        self.assertEqual(RuntimeConfig().env_file, ".env.local")
        self.assertEqual(self.config.runtime.env_file, ".env.local")

    def test_env_reader_accepts_export_and_quoted_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env.local"
            env_path.write_text(
                "# Local W&B credentials\n"
                "export WANDB_API_KEY='fake-test-value'\n"
                'WANDB_MODE="online"\n',
                encoding="utf-8",
            )

            values = launcher._read_env_file(str(env_path))

        self.assertEqual(values["WANDB_API_KEY"], "fake-test-value")
        self.assertEqual(values["WANDB_MODE"], "online")

    def test_relative_env_file_is_resolved_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir) / "project"
            fake_launcher = project_root / "mopd_verl/launch.py"
            fake_launcher.parent.mkdir(parents=True)
            (project_root / ".env.local").write_text(
                "WANDB_API_KEY=fake-project-root-value\n",
                encoding="utf-8",
            )

            with patch.object(launcher, "__file__", str(fake_launcher)):
                values = launcher._read_env_file(".env.local")

        self.assertEqual(values["WANDB_API_KEY"], "fake-project-root-value")

    def test_training_subprocess_receives_key_without_overriding_shell(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env.local"
            env_path.write_text(
                "export WANDB_API_KEY=fake-test-value\n"
                "export WANDB_MODE=online\n"
                "export WANDB_ENTITY=env-file-team\n",
                encoding="utf-8",
            )
            config = replace(
                self.config,
                runtime=replace(
                    self.config.runtime,
                    env_file=str(env_path),
                    wandb_entity="yaml-team",
                ),
            )

            with patch.dict(os.environ, {"WANDB_MODE": "offline"}, clear=True):
                with patch.object(launcher.subprocess, "call", return_value=0) as call:
                    return_code = launcher.run_command(["python3", "train.py"], config)

            child_env = call.call_args.kwargs["env"]
            self.assertEqual(return_code, 0)
            self.assertEqual(child_env["WANDB_API_KEY"], "fake-test-value")
            self.assertEqual(child_env["WANDB_MODE"], "offline")
            self.assertEqual(child_env["WANDB_ENTITY"], "env-file-team")

    def test_remote_sync_excludes_secret_env_files(self) -> None:
        sync_script = (self.project_root / "scripts/sync_remote.sh").read_text(
            encoding="utf-8"
        )

        example_include = sync_script.index('--include "/.env.local.example"')
        secret_exclude = sync_script.index('--exclude ".env.*"')
        self.assertLess(example_include, secret_exclude)
        self.assertIn('--exclude ".env"', sync_script)

    def test_tracking_resumes_explicit_wandb_run_id(self) -> None:
        tracking_path = (
            self.project_root / "third_party/verl/verl/utils/tracking.py"
        )
        spec = importlib.util.spec_from_file_location("vendored_verl_tracking", tracking_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        tracking_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tracking_module)
        fake_wandb = MagicMock()
        with (
            patch.dict(
                os.environ,
                {"WANDB_RUN_ID": "stable-test-run", "WANDB_RESUME": "must"},
                clear=True,
            ),
            patch.dict(sys.modules, {"wandb": fake_wandb}),
        ):
            tracker = tracking_module.Tracking(
                "project", "experiment", default_backend=["wandb"]
            )
            del tracker

        init_kwargs = fake_wandb.init.call_args.kwargs
        self.assertEqual(init_kwargs["id"], "stable-test-run")
        self.assertEqual(init_kwargs["resume"], "must")


if __name__ == "__main__":
    unittest.main()
