#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SSH_CONFIG="${GRPO_SSH_CONFIG:-${SCRIPT_DIR}/ssh.sh}"

if [[ ! -f "${SSH_CONFIG}" ]]; then
  echo "Missing SSH config: ${SSH_CONFIG}" >&2
  echo "Copy scripts/ssh.example.sh to scripts/ssh.sh and fill the remote settings." >&2
  exit 2
fi

REMOTE_HOST_OVERRIDE="${REMOTE_HOST:-}"
REMOTE_USER_OVERRIDE="${REMOTE_USER:-}"
REMOTE_PORT_OVERRIDE="${REMOTE_PORT:-}"
REMOTE_PROJECT_DIR_OVERRIDE="${REMOTE_PROJECT_DIR:-}"
REMOTE_CONDA_ENV_OVERRIDE="${REMOTE_CONDA_ENV:-}"
REMOTE_PYTHON_VERSION_OVERRIDE="${REMOTE_PYTHON_VERSION:-}"
SSH_KEY_OVERRIDE="${SSH_KEY:-}"

# shellcheck source=/dev/null
source "${SSH_CONFIG}"

REMOTE_HOST="${REMOTE_HOST_OVERRIDE:-${REMOTE_HOST:-}}"
REMOTE_USER="${REMOTE_USER_OVERRIDE:-${REMOTE_USER:-root}}"
REMOTE_PORT="${REMOTE_PORT_OVERRIDE:-${REMOTE_PORT:-22}}"
REMOTE_PROJECT_DIR="${REMOTE_PROJECT_DIR_OVERRIDE:-${REMOTE_PROJECT_DIR:-/root/autodl-tmp/GRPO}}"
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV_OVERRIDE:-${REMOTE_CONDA_ENV:-grpo-rl}}"
REMOTE_PYTHON_VERSION="${REMOTE_PYTHON_VERSION_OVERRIDE:-${REMOTE_PYTHON_VERSION:-3.10}}"
SSH_KEY="${SSH_KEY_OVERRIDE:-${SSH_KEY:-}}"

if [[ -z "${REMOTE_HOST}" ]]; then
  echo "REMOTE_HOST is empty. Edit ${SSH_CONFIG} first." >&2
  exit 2
fi

REMOTE_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
SSH_ARGS=(-p "${REMOTE_PORT}")
if [[ -n "${SSH_KEY}" ]]; then
  SSH_ARGS+=(-i "${SSH_KEY}")
fi
if declare -p SSH_OPTIONS >/dev/null 2>&1; then
  SSH_ARGS+=("${SSH_OPTIONS[@]}")
fi

SSH_CMD=(ssh)
if [[ -n "${SSHPASS:-}" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "SSHPASS is set, but sshpass is not installed locally." >&2
    exit 2
  fi
  SSH_CMD=(sshpass -e ssh)
fi

RSYNC_RSH=("${SSH_CMD[@]}" "${SSH_ARGS[@]}")
RSYNC_RSH_STRING="$(printf " %q" "${RSYNC_RSH[@]}")"
RSYNC_RSH_STRING="${RSYNC_RSH_STRING# }"

remote_bash() {
  local command="$1"
  "${SSH_CMD[@]}" "${SSH_ARGS[@]}" "${REMOTE_TARGET}" "bash -lc $(printf "%q" "${command}")"
}
