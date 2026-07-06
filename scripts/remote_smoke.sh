#!/usr/bin/env bash
set -euo pipefail

LOCAL_MODE=0
if [[ "${1:-}" == "--local" ]]; then
  LOCAL_MODE=1
  shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${LOCAL_MODE}" == "0" ]]; then
  # shellcheck source=scripts/remote_common.sh
  source "${SCRIPT_DIR}/remote_common.sh"
  REMOTE_SMOKE_CONFIG="${REMOTE_SMOKE_CONFIG:-grpo/configs/m2rl_if_smoke.yaml}"
  REMOTE_SMOKE_DRY_RUN="${REMOTE_SMOKE_DRY_RUN:-1}"
  REMOTE_SMOKE_OVERRIDES="${REMOTE_SMOKE_OVERRIDES:-}"
  remote_bash "cd $(printf "%q" "${REMOTE_PROJECT_DIR}") && REMOTE_CONDA_ENV=$(printf "%q" "${REMOTE_CONDA_ENV}") REMOTE_SMOKE_CONFIG=$(printf "%q" "${REMOTE_SMOKE_CONFIG}") REMOTE_SMOKE_DRY_RUN=$(printf "%q" "${REMOTE_SMOKE_DRY_RUN}") REMOTE_SMOKE_OVERRIDES=$(printf "%q" "${REMOTE_SMOKE_OVERRIDES}") bash scripts/remote_smoke.sh --local"
  exit $?
fi

ENV_NAME="${REMOTE_CONDA_ENV:-${CONDA_ENV_NAME:-grpo-rl}}"
SMOKE_CONFIG="${REMOTE_SMOKE_CONFIG:-grpo/configs/m2rl_if_smoke.yaml}"
SMOKE_DRY_RUN="${REMOTE_SMOKE_DRY_RUN:-1}"
SMOKE_OVERRIDES="${REMOTE_SMOKE_OVERRIDES:-}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  for candidate in "${HOME}/miniconda3/etc/profile.d/conda.sh" "${HOME}/anaconda3/etc/profile.d/conda.sh" "/opt/conda/etc/profile.d/conda.sh" "/root/miniconda3/etc/profile.d/conda.sh"; do
    if [[ -f "${candidate}" ]]; then
      # shellcheck source=/dev/null
      source "${candidate}"
      conda activate "${ENV_NAME}"
      break
    fi
  done
fi

export IFBENCH_REPO="${IFBENCH_REPO:-${PWD}/IFBench}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONPATH="${PWD}:${PWD}/third_party/verl:${PYTHONPATH:-}"

ARGS=("${SMOKE_CONFIG}")
if [[ "${SMOKE_DRY_RUN}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

if [[ -n "${SMOKE_OVERRIDES}" ]]; then
  # Intentional word splitting: this variable is an operator-provided override string.
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${SMOKE_OVERRIDES})
  ARGS+=(-- "${EXTRA_ARGS[@]}")
fi

scripts/run_grpo.sh "${ARGS[@]}"

