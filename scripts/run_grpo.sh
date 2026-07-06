#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_grpo.sh [config] [--dry-run] [-- <hydra overrides...>]

Examples:
  scripts/run_grpo.sh grpo/configs/m2rl_if_science_mix.yaml --dry-run

  scripts/run_grpo.sh grpo/configs/m2rl_science.yaml -- \
    actor_rollout_ref.model.path=../models/Qwen3-4B

Environment:
  GRPO_CONFIG=<default config when config arg is omitted>
  VERL_RUNTIME_DIR=<vendored verl runtime dir>
  PYTHON=<python executable, defaults to python3>
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_CONFIG="${PROJECT_DIR}/grpo/configs/m2rl_if_science_mix.yaml"
CONFIG_PATH="${GRPO_CONFIG:-${DEFAULT_CONFIG}}"
VERL_RUNTIME_DIR="${VERL_RUNTIME_DIR:-${PROJECT_DIR}/third_party/verl}"
PYTHON_BIN="${PYTHON:-python3}"
DRY_RUN_FLAG=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN_FLAG=1
      shift
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    -*)
      echo "Unknown script option: $1" >&2
      echo "Put Hydra overrides after '--'." >&2
      exit 2
      ;;
    *)
      if [[ "${CONFIG_PATH}" != "${GRPO_CONFIG:-${DEFAULT_CONFIG}}" ]]; then
        echo "Only one config path is allowed." >&2
        exit 2
      fi
      CONFIG_PATH="$1"
      shift
      ;;
  esac
done

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${PROJECT_DIR}/${CONFIG_PATH}"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

if [[ ! -f "${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py" ]]; then
  echo "Vendored verl runtime not found at '${VERL_RUNTIME_DIR}'." >&2
  echo "Expected '${VERL_RUNTIME_DIR}/verl/trainer/main_ppo.py'." >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_DIR}:${VERL_RUNTIME_DIR}:${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS="${PYTHONINTMAXSTRDIGITS:-0}"
cd "${PROJECT_DIR}"

ARGS=(--config "${CONFIG_PATH}")
if [[ "${DRY_RUN:-0}" == "1" || "${DRY_RUN_FLAG}" == "1" ]]; then
  ARGS+=(--dry-run)
fi
if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  ARGS+=(-- "${EXTRA_ARGS[@]}")
fi

exec "${PYTHON_BIN}" -m grpo_training.launch "${ARGS[@]}"
