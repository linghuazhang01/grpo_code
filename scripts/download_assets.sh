#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${REMOTE_CONDA_ENV:-${CONDA_ENV_NAME:-grpo-rl}}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"
MODEL_IDS="${MODEL_IDS:-Qwen/Qwen3-4B}"
MODEL_DIR="${MODEL_DIR:-models}"
IFBENCH_DIR="${IFBENCH_DIR:-IFBench}"
IFBENCH_REPO_URL="${IFBENCH_REPO_URL:-https://github.com/allenai/IFBench.git}"
INSTALL_IFBENCH_DEPS="${INSTALL_IFBENCH_DEPS:-0}"
DOWNLOAD_IFBENCH_NLTK="${DOWNLOAD_IFBENCH_NLTK:-${INSTALL_IFBENCH_DEPS}}"
CREATE_SMOKE_DATA="${CREATE_SMOKE_DATA:-0}"

activate_conda_if_available() {
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    return 0
  fi
  local candidates=(
    "${HOME}/miniconda3/etc/profile.d/conda.sh"
    "${HOME}/anaconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
    "/root/miniconda3/etc/profile.d/conda.sh"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      # shellcheck source=/dev/null
      source "${candidate}"
      if [[ -d "${ENV_NAME}" ]] || conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
        conda activate "${ENV_NAME}"
        return 0
      fi
      echo "Conda environment not found, continuing without activation: ${ENV_NAME}" >&2
      return 1
    fi
  done
}

activate_conda_if_available || true

mkdir -p "${MODEL_DIR}" data/M2RL/if data/M2RL/science \
  eval/domains/ifbench/data eval/domains/science/data

if [[ "${DOWNLOAD_MODELS}" == "1" ]]; then
  if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
    python -m pip install huggingface_hub
  fi
  for model_id in ${MODEL_IDS}; do
    local_name="${model_id##*/}"
    echo "Downloading model ${model_id} -> ${MODEL_DIR}/${local_name}"
    if command -v hf >/dev/null 2>&1; then
      hf download "${model_id}" --local-dir "${MODEL_DIR}/${local_name}"
    else
      huggingface-cli download "${model_id}" --local-dir "${MODEL_DIR}/${local_name}"
    fi
  done
fi

if [[ ! -d "${IFBENCH_DIR}/.git" ]]; then
  echo "Cloning IFBench -> ${IFBENCH_DIR}"
  git -c http.version=HTTP/1.1 clone "${IFBENCH_REPO_URL}" "${IFBENCH_DIR}"
else
  echo "IFBench already exists: ${IFBENCH_DIR}"
fi

if [[ "${INSTALL_IFBENCH_DEPS}" == "1" && -f "${IFBENCH_DIR}/requirements.txt" ]]; then
  python -m pip install -r "${IFBENCH_DIR}/requirements.txt"
fi

all_nltk_targets_exist() {
  [[ -d "${IFBENCH_DIR}/.nltk_data/tokenizers/punkt" ]] &&
    [[ -d "${IFBENCH_DIR}/.nltk_data/tokenizers/punkt_tab" ]] &&
    [[ -d "${IFBENCH_DIR}/.nltk_data/corpora/stopwords" ]] &&
    [[ -d "${IFBENCH_DIR}/.nltk_data/taggers/averaged_perceptron_tagger_eng" ]]
}

download_nltk_with_git() {
  all_nltk_targets_exist && return 0
  if ! command -v git >/dev/null 2>&1; then
    return 1
  fi

  local tmp_dir
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "${tmp_dir}"' RETURN
  git -c http.version=HTTP/1.1 clone --depth=1 --filter=blob:none --sparse https://github.com/nltk/nltk_data.git "${tmp_dir}/nltk_data"
  (
    cd "${tmp_dir}/nltk_data"
    git sparse-checkout set --no-cone \
      packages/tokenizers/punkt.zip \
      packages/tokenizers/punkt_tab.zip \
      packages/corpora/stopwords.zip \
      packages/taggers/averaged_perceptron_tagger_eng.zip
  )
  mkdir -p "${IFBENCH_DIR}/.nltk_data/tokenizers" \
    "${IFBENCH_DIR}/.nltk_data/corpora" \
    "${IFBENCH_DIR}/.nltk_data/taggers"
  python -m zipfile -e "${tmp_dir}/nltk_data/packages/tokenizers/punkt.zip" "${IFBENCH_DIR}/.nltk_data/tokenizers"
  python -m zipfile -e "${tmp_dir}/nltk_data/packages/tokenizers/punkt_tab.zip" "${IFBENCH_DIR}/.nltk_data/tokenizers"
  python -m zipfile -e "${tmp_dir}/nltk_data/packages/corpora/stopwords.zip" "${IFBENCH_DIR}/.nltk_data/corpora"
  python -m zipfile -e "${tmp_dir}/nltk_data/packages/taggers/averaged_perceptron_tagger_eng.zip" "${IFBENCH_DIR}/.nltk_data/taggers"
}

download_nltk_zip() {
  local package_group="$1"
  local package_name="$2"
  local target_dir="${IFBENCH_DIR}/.nltk_data/${package_group}"
  local target_path="${target_dir}/${package_name}"
  local archive_path="${IFBENCH_DIR}/.nltk_data/${package_name}.zip"

  if [[ -e "${target_path}" ]]; then
    return 0
  fi

  mkdir -p "${target_dir}"
  curl -L -C - --fail --retry 3 --retry-delay 5 --connect-timeout 20 --max-time 600 \
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/${package_group}/${package_name}.zip" \
    -o "${archive_path}"
  python -m zipfile -e "${archive_path}" "${target_dir}"
  rm -f "${archive_path}"
}

if [[ "${DOWNLOAD_IFBENCH_NLTK}" == "1" ]]; then
  if ! download_nltk_with_git; then
    download_nltk_zip tokenizers punkt
    download_nltk_zip tokenizers punkt_tab
    download_nltk_zip corpora stopwords
    download_nltk_zip taggers averaged_perceptron_tagger_eng
  fi
fi

if [[ "${CREATE_SMOKE_DATA}" == "1" ]]; then
  export PYTHONPATH="${PWD}:${PWD}/third_party/verl:${PYTHONPATH:-}"
  python - <<'PY'
from pathlib import Path

import pandas as pd

from grpo.data.m2rl import m2rl_frame_to_verl

raw = pd.DataFrame(
    [
        {
            "prompt": "Write exactly two words.",
            "label": "",
            "metadata": {
                "instruction_id_list": ["count:word_count_range"],
                "kwargs": [{"min_words": 2, "max_words": 2}],
                "prompt_text": "Write exactly two words.",
                "record_id": 0,
            },
        },
        {
            "prompt": "Write exactly three words.",
            "label": "",
            "metadata": {
                "instruction_id_list": ["count:word_count_range"],
                "kwargs": [{"min_words": 3, "max_words": 3}],
                "prompt_text": "Write exactly three words.",
                "record_id": 1,
            },
        },
    ]
)
normalized = m2rl_frame_to_verl(raw, rm_type="ifbench", split="smoke", domain="if")
paths = [
    Path("data/M2RL/if/train.parquet"),
    Path("eval/domains/ifbench/data/IFBench_test.parquet"),
]
for path in paths:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(path, index=False)
    print(f"Wrote smoke data: {path}")
PY
fi

prepare_m2rl_if_source() {
  local source_path="$1"
  local output_path="$2"
  local split="$3"
  if [[ -z "${source_path}" ]]; then
    return 0
  fi
  if [[ ! -f "${source_path}" ]]; then
    echo "Missing IF source file: ${source_path}" >&2
    return 1
  fi
  python -m grpo_training.prepare_data prepare-m2rl \
    --input "${source_path}" \
    --output "${output_path}" \
    --rm-type ifbench \
    --domain if \
    --split "${split}"
}

prepare_m2rl_science_source() {
  local source_path="$1"
  local output_path="$2"
  local split="$3"
  if [[ -z "${source_path}" ]]; then
    return 0
  fi
  if [[ ! -f "${source_path}" ]]; then
    echo "Missing science source file: ${source_path}" >&2
    return 1
  fi
  python -m grpo_training.prepare_data prepare-m2rl \
    --input "${source_path}" \
    --output "${output_path}" \
    --rm-type gpqa \
    --domain science \
    --split "${split}"
}

prepare_m2rl_if_source "${IF_TRAIN_SOURCE:-}" "data/M2RL/if/train.parquet" "train"
prepare_m2rl_if_source "${IF_VAL_SOURCE:-}" "eval/domains/ifbench/data/IFBench_test.parquet" "validation"
prepare_m2rl_science_source "${SCIENCE_TRAIN_SOURCE:-}" "data/M2RL/science/train.parquet" "train"
prepare_m2rl_science_source "${SCIENCE_VAL_SOURCE:-}" "eval/domains/science/data/gpqa.parquet" "validation"

cat <<'EOF'
Asset setup finished.

If no data source variables were provided, prepare these files before real training:
  data/M2RL/if/train.parquet
  data/M2RL/science/train.parquet
  eval/domains/ifbench/data/IFBench_test.parquet
  eval/domains/science/data/gpqa.parquet

Supported source variables:
  IF_TRAIN_SOURCE=/path/to/raw_if_train.parquet
  IF_VAL_SOURCE=/path/to/raw_if_val.parquet
  SCIENCE_TRAIN_SOURCE=/path/to/raw_science_train.parquet
  SCIENCE_VAL_SOURCE=/path/to/raw_science_val.parquet
EOF
