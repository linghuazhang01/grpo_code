#!/usr/bin/env bash
# Copy or edit scripts/ssh.sh with your real remote settings.
# Do not commit secrets, private keys, tokens, or passwords.

REMOTE_HOST=""
REMOTE_USER="root"
REMOTE_PORT="22"
REMOTE_PROJECT_DIR="/root/autodl-tmp/GRPO"
REMOTE_CONDA_ENV="grpo-rl"
REMOTE_PYTHON_VERSION="3.10"

# Optional private key path. Leave empty to use your default SSH agent/config.
SSH_KEY=""

# Extra SSH options. Keep this as a bash array.
SSH_OPTIONS=(
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
)

