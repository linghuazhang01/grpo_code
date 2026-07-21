#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_m2rl_141gb.sh <science|if> <6|8> [--dry-run] [-- <hydra overrides...>]

Environment overrides:
  GRPO_MODEL_PATH       Model path (default: /root/OPD/models/Qwen3-4B)
  SCIENCE_TRAIN_FILE    Science training parquet
  SCIENCE_VAL_FILE      Science validation parquet
  IF_TRAIN_FILE         IF training parquet
  IF_VAL_FILE           IF validation parquet
  GRPO_CHECKPOINT_DIR   Checkpoint output directory
  CUDA_VISIBLE_DEVICES  Explicit GPU selection; defaults to the first 6 or 8 GPUs
  GRPO_SKIP_PREFLIGHT   Set to 1 only for Hydra composition/debugging without assets

The profile uses a 16K maximum response, 16 rollouts per prompt, saves every
10 steps, keeps the latest 5 checkpoints, and stops at 1200 training steps.
The logical prompt batch is 126 on 6 GPUs and 128 on 8 GPUs.
USAGE
}

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

DOMAIN="$1"
GPU_COUNT="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_PATH="${GRPO_MODEL_PATH:-/root/OPD/models/Qwen3-4B}"
DRY_RUN_FLAG=0
EXTRA_OVERRIDES=()

case "${DOMAIN}" in
  science)
    TRAIN_FILE="${SCIENCE_TRAIN_FILE:-data/M2RL/science/train.parquet}"
    VAL_FILE="${SCIENCE_VAL_FILE:-eval/domains/science/data/gpqa.parquet}"
    EXPERIMENT_BASE="qwen4b-m2rl-science"
    ;;
  if)
    TRAIN_FILE="${IF_TRAIN_FILE:-data/M2RL/if/train.parquet}"
    VAL_FILE="${IF_VAL_FILE:-eval/domains/ifbench/data/IFBench_test.parquet}"
    EXPERIMENT_BASE="qwen4b-m2rl-if"
    ;;
  *)
    echo "Unsupported domain: ${DOMAIN}; expected science or if." >&2
    exit 2
    ;;
esac

case "${GPU_COUNT}" in
  6)
    DEFAULT_VISIBLE_DEVICES="0,1,2,3,4,5"
    ;;
  8)
    DEFAULT_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    ;;
  *)
    echo "Unsupported GPU count: ${GPU_COUNT}; expected 6 or 8." >&2
    exit 2
    ;;
esac

CONFIG_PATH="${PROJECT_DIR}/grpo/configs/m2rl_${DOMAIN}_${GPU_COUNT}gpu_141gb.yaml"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Training profile not found: ${CONFIG_PATH}" >&2
  exit 2
fi

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
      EXTRA_OVERRIDES=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Put Hydra overrides after '--'." >&2
      exit 2
      ;;
  esac
done

for override in "${EXTRA_OVERRIDES[@]}"; do
  override_key="${override%%=*}"
  while [[ "${override_key}" == +* ]]; do
    override_key="${override_key#+}"
  done
  override_key="${override_key#\~}"
  case "${override_key}" in
    data|data.apply_chat_template_kwargs|data.apply_chat_template_kwargs.enable_thinking|\
    actor_rollout_ref|actor_rollout_ref.model|actor_rollout_ref.model.override_config|\
    actor_rollout_ref.model.override_config.attn_implementation|actor_rollout_ref.model.use_remove_padding|\
    actor_rollout_ref.ref|actor_rollout_ref.ref.model|\
    actor_rollout_ref.actor|actor_rollout_ref.actor.fsdp_config|actor_rollout_ref.rollout|trainer|\
    actor_rollout_ref.actor.policy_loss|actor_rollout_ref.actor.policy_loss.*|\
    data.train_files|data.val_files|data.train_batch_size|data.val_batch_size|data.max_prompt_length|data.max_response_length|\
    actor_rollout_ref.model.path|actor_rollout_ref.model.base_model_path|actor_rollout_ref.ref.model.path|\
    actor_rollout_ref.actor.ppo_mini_batch_size|actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu|\
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu|actor_rollout_ref.actor.use_dynamic_bsz|\
    actor_rollout_ref.actor.use_kl_loss|actor_rollout_ref.actor.kl_loss_coef|\
    actor_rollout_ref.actor.entropy_coeff|\
    actor_rollout_ref.actor.fsdp_config.model_dtype|actor_rollout_ref.rollout.tensor_model_parallel_size|\
    actor_rollout_ref.rollout.n|actor_rollout_ref.rollout.max_model_len|\
    actor_rollout_ref.rollout.max_num_batched_tokens|actor_rollout_ref.rollout.max_num_seqs|\
    trainer.n_gpus_per_node|trainer.nnodes|trainer.save_freq|trainer.max_actor_ckpt_to_keep|\
    trainer.max_critic_ckpt_to_keep|trainer.total_training_steps|trainer.default_local_dir|\
    trainer.experiment_name|algorithm|algorithm.*)
      echo "Protected 141GB profile setting cannot be overridden: ${override_key}" >&2
      echo "Use the documented GRPO_MODEL_PATH, *_TRAIN_FILE, *_VAL_FILE, or GRPO_CHECKPOINT_DIR variables." >&2
      exit 2
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_VISIBLE_DEVICES}}"

IFS=',' read -r -a VISIBLE_DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
if [[ "${#VISIBLE_DEVICES[@]}" -ne "${GPU_COUNT}" ]]; then
  echo "CUDA_VISIBLE_DEVICES exposes ${#VISIBLE_DEVICES[@]} GPUs, but the ${GPU_COUNT}-GPU profile was requested." >&2
  exit 2
fi

CHECKPOINT_DIR="${GRPO_CHECKPOINT_DIR:-checkpoints/${EXPERIMENT_BASE}-${GPU_COUNT}gpu-141gb-16k}"

if [[ "${DRY_RUN_FLAG}" != "1" && "${GRPO_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  ACTUAL_GPU_COUNT="$("${PYTHON:-python3}" -c 'import torch; print(torch.cuda.device_count())')"
  if [[ "${ACTUAL_GPU_COUNT}" != "${GPU_COUNT}" ]]; then
    echo "PyTorch sees ${ACTUAL_GPU_COUNT} GPUs after CUDA_VISIBLE_DEVICES, but ${GPU_COUNT} are required." >&2
    exit 2
  fi

  TRAIN_CHECK_PATH="${TRAIN_FILE}"
  VAL_CHECK_PATH="${VAL_FILE}"
  CHECKPOINT_ROOT="${CHECKPOINT_DIR}"
  if [[ "${TRAIN_CHECK_PATH}" != /* ]]; then
    TRAIN_CHECK_PATH="${PROJECT_DIR}/${TRAIN_CHECK_PATH}"
  fi
  if [[ "${VAL_CHECK_PATH}" != /* ]]; then
    VAL_CHECK_PATH="${PROJECT_DIR}/${VAL_CHECK_PATH}"
  fi
  if [[ "${CHECKPOINT_ROOT}" != /* ]]; then
    CHECKPOINT_ROOT="${PROJECT_DIR}/${CHECKPOINT_ROOT}"
  fi
  for required_file in "${TRAIN_CHECK_PATH}" "${VAL_CHECK_PATH}"; do
    if [[ ! -f "${required_file}" ]]; then
      echo "Required data file not found: ${required_file}" >&2
      echo "Set the corresponding ${DOMAIN^^}_TRAIN_FILE/${DOMAIN^^}_VAL_FILE variable." >&2
      exit 2
    fi
  done
  MODEL_CHECK_PATH="${MODEL_PATH}"
  if [[ "${MODEL_CHECK_PATH}" != /* ]]; then
    MODEL_CHECK_PATH="${PROJECT_DIR}/${MODEL_CHECK_PATH}"
  fi
  if [[ ! -f "${MODEL_CHECK_PATH}/config.json" ]]; then
    echo "Model config not found: ${MODEL_CHECK_PATH}/config.json" >&2
    exit 2
  fi

  export PYTHONPATH="${PROJECT_DIR}:${PROJECT_DIR}/third_party/verl:${PYTHONPATH:-}"
  RM_TYPE="gpqa"
  if [[ "${DOMAIN}" == "if" ]]; then
    RM_TYPE="ifbench"
  fi
  "${PYTHON:-python3}" -m grpo.data.m2rl validate --input "${TRAIN_CHECK_PATH}" --rm-type "${RM_TYPE}"
  "${PYTHON:-python3}" -m grpo.data.m2rl validate --input "${VAL_CHECK_PATH}" --rm-type "${RM_TYPE}"
  if [[ "${DOMAIN}" == "if" ]]; then
    "${PYTHON:-python3}" -c \
      "from grpo.rewards.m2rl import _ensure_ifbench_importable; _ensure_ifbench_importable()"
  fi

  mkdir -p "${CHECKPOINT_ROOT}"
  WANDB_RUN_ID_FILE="${CHECKPOINT_ROOT}/.wandb_run_id"
  if [[ -z "${WANDB_RUN_ID:-}" && -f "${WANDB_RUN_ID_FILE}" ]]; then
    WANDB_RUN_ID="$(<"${WANDB_RUN_ID_FILE}")"
  fi
  if [[ -z "${WANDB_RUN_ID:-}" ]]; then
    WANDB_RUN_ID="$("${PYTHON:-python3}" -c 'import secrets; print(secrets.token_hex(16))')"
    (umask 077 && printf '%s\n' "${WANDB_RUN_ID}" > "${WANDB_RUN_ID_FILE}")
  fi
  export WANDB_RUN_ID
  export WANDB_RESUME="${WANDB_RESUME:-allow}"
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
RUN_ARGS+=(-- "${PROFILE_OVERRIDES[@]}" "${EXTRA_OVERRIDES[@]}")

exec "${SCRIPT_DIR}/run_grpo.sh" "${RUN_ARGS[@]}"
