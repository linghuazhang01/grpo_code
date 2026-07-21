#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/remote_common.sh
source "${SCRIPT_DIR}/remote_common.sh"

RSYNC_DELETE="${RSYNC_DELETE:-0}"
SYNC_DATA="${SYNC_DATA:-0}"
RUN_REMOTE_INSTALL="${RUN_REMOTE_INSTALL:-0}"
RUN_REMOTE_DOWNLOAD="${RUN_REMOTE_DOWNLOAD:-0}"
RUN_REMOTE_SMOKE="${RUN_REMOTE_SMOKE:-0}"

RSYNC_ARGS=(-az)
if rsync --help 2>&1 | grep -q -- "--info"; then
  RSYNC_ARGS+=(--info=progress2)
else
  RSYNC_ARGS+=(--progress)
fi
if [[ "${RSYNC_DELETE}" == "1" ]]; then
  RSYNC_ARGS+=(--delete)
fi

RSYNC_EXCLUDES=(
  --exclude ".git/"
  --exclude ".venv/"
  --include "/.env.example"
  --include "/.env.local.example"
  --exclude ".env"
  --exclude ".env.*"
  --exclude "__pycache__/"
  --exclude "*.pyc"
  --exclude ".DS_Store"
  --exclude "checkpoints/***"
  --exclude "logs/***"
  --exclude "wandb/***"
  --exclude "audit/***"
  --exclude "temp/***"
  --exclude "/models/***"
  --exclude "/IFBench/***"
  --exclude "*.safetensors"
  --exclude "*.pt"
  --exclude "*.pth"
  --exclude "*.ckpt"
)

if [[ "${SYNC_DATA}" != "1" ]]; then
  RSYNC_EXCLUDES+=(--exclude "/data/***" --exclude "/eval/**/*.parquet")
fi

remote_bash "mkdir -p $(printf "%q" "${REMOTE_PROJECT_DIR}")"

rsync "${RSYNC_ARGS[@]}" "${RSYNC_EXCLUDES[@]}" \
  -e "${RSYNC_RSH_STRING}" \
  "${PROJECT_DIR}/" \
  "${REMOTE_TARGET}:${REMOTE_PROJECT_DIR}/"

if [[ "${RUN_REMOTE_INSTALL}" == "1" ]]; then
  remote_bash "cd $(printf "%q" "${REMOTE_PROJECT_DIR}") && REMOTE_CONDA_ENV=$(printf "%q" "${REMOTE_CONDA_ENV}") REMOTE_PYTHON_VERSION=$(printf "%q" "${REMOTE_PYTHON_VERSION}") bash scripts/install_conda_env.sh"
fi

if [[ "${RUN_REMOTE_DOWNLOAD}" == "1" ]]; then
  remote_bash "cd $(printf "%q" "${REMOTE_PROJECT_DIR}") && REMOTE_CONDA_ENV=$(printf "%q" "${REMOTE_CONDA_ENV}") bash scripts/download_assets.sh"
fi

if [[ "${RUN_REMOTE_SMOKE}" == "1" ]]; then
  remote_bash "cd $(printf "%q" "${REMOTE_PROJECT_DIR}") && REMOTE_CONDA_ENV=$(printf "%q" "${REMOTE_CONDA_ENV}") bash scripts/remote_smoke.sh --local"
fi
