#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
General-Reasoner GRPO was archived when this repository was switched to M2RL IF/Science.
Use one of:
  scripts/run_m2rl_if_grpo.sh
  scripts/run_m2rl_science_grpo.sh
  scripts/run_m2rl_if_science_grpo.sh

The legacy grpo/ package backup path is recorded in temp/latest_grpo_legacy_backup.txt.
EOF
exit 2
