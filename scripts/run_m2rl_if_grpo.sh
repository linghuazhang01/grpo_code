#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${GRPO_CONFIG:-${PROJECT_DIR}/grpo/configs/m2rl_if.yaml}"

exec env GRPO_CONFIG="${CONFIG_PATH}" "${SCRIPT_DIR}/run_grpo.sh" "$@"

