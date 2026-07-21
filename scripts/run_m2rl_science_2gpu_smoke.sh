#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_m2rl_science_2gpu_smoke.sh [--dry-run]

Environment overrides:
  GRPO_MODEL_PATH       Model path (default: /root/OPD/models/Qwen3-4B)
  SCIENCE_TRAIN_FILE    Science training parquet
  SCIENCE_VAL_FILE      Science validation parquet
  GRPO_CHECKPOINT_DIR   Smoke output directory
  CUDA_VISIBLE_DEVICES  Two GPU indices (default: 0,1)
  GRPO_SKIP_PREFLIGHT   Set to 1 only for composition/debugging without assets
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${PROJECT_DIR}/grpo/configs/m2rl_science_smoke_2gpu.yaml"
MODEL_PATH="${GRPO_MODEL_PATH:-/root/OPD/models/Qwen3-4B}"
TRAIN_FILE="${SCIENCE_TRAIN_FILE:-data/M2RL/science/train.parquet}"
VAL_FILE="${SCIENCE_VAL_FILE:-eval/domains/science/data/gpqa.parquet}"
CHECKPOINT_DIR="${GRPO_CHECKPOINT_DIR:-checkpoints/qwen4b-m2rl-science-2gpu-smoke}"
DRY_RUN_FLAG=0

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
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a VISIBLE_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#VISIBLE_DEVICES[@]}" -ne 2 ]]; then
  echo "CUDA_VISIBLE_DEVICES must expose exactly 2 GPUs for this smoke profile." >&2
  exit 2
fi

if [[ "${DRY_RUN_FLAG}" != "1" && "${GRPO_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  ACTUAL_GPU_COUNT="$("${PYTHON:-python3}" -c 'import torch; print(torch.cuda.device_count())')"
  if [[ "${ACTUAL_GPU_COUNT}" != "2" ]]; then
    echo "PyTorch sees ${ACTUAL_GPU_COUNT} GPUs after CUDA_VISIBLE_DEVICES; 2 are required." >&2
    exit 2
  fi

  TRAIN_CHECK_PATH="${TRAIN_FILE}"
  VAL_CHECK_PATH="${VAL_FILE}"
  MODEL_CHECK_PATH="${MODEL_PATH}"
  if [[ "${TRAIN_CHECK_PATH}" != /* ]]; then
    TRAIN_CHECK_PATH="${PROJECT_DIR}/${TRAIN_CHECK_PATH}"
  fi
  if [[ "${VAL_CHECK_PATH}" != /* ]]; then
    VAL_CHECK_PATH="${PROJECT_DIR}/${VAL_CHECK_PATH}"
  fi
  if [[ "${MODEL_CHECK_PATH}" != /* ]]; then
    MODEL_CHECK_PATH="${PROJECT_DIR}/${MODEL_CHECK_PATH}"
  fi

  for required_file in "${TRAIN_CHECK_PATH}" "${VAL_CHECK_PATH}"; do
    if [[ ! -f "${required_file}" ]]; then
      echo "Required data file not found: ${required_file}" >&2
      exit 2
    fi
  done
  if [[ ! -f "${MODEL_CHECK_PATH}/config.json" ]]; then
    echo "Model config not found: ${MODEL_CHECK_PATH}/config.json" >&2
    exit 2
  fi

  export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/third_party/verl:${PYTHONPATH:-}"
  "${PYTHON:-python3}" -m grpo.data.m2rl validate \
    --input "${TRAIN_CHECK_PATH}" --rm-type gpqa
  "${PYTHON:-python3}" -m grpo.data.m2rl validate \
    --input "${VAL_CHECK_PATH}" --rm-type gpqa
fi

PROFILE_OVERRIDES=(
  "data.train_files=[\"${TRAIN_FILE}\"]"
  "data.val_files=[\"${VAL_FILE}\"]"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.base_model_path=${MODEL_PATH}"
  "actor_rollout_ref.ref.model.path=${MODEL_PATH}"
  "trainer.default_local_dir=${CHECKPOINT_DIR}"
)

RUN_ARGS=("${CONFIG_PATH}")
if [[ "${DRY_RUN_FLAG}" == "1" ]]; then
  RUN_ARGS+=(--dry-run)
fi
RUN_ARGS+=(-- "${PROFILE_OVERRIDES[@]}")

exec "${SCRIPT_DIR}/run_grpo.sh" "${RUN_ARGS[@]}"
