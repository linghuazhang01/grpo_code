#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${REMOTE_CONDA_ENV:-${CONDA_ENV_NAME:-grpo-rl}}"
PYTHON_VERSION="${REMOTE_PYTHON_VERSION:-${PYTHON_VERSION:-3.10}}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
VLLM_SPEC="${VLLM_SPEC:-vllm==0.8.4}"

find_conda() {
  if command -v conda >/dev/null 2>&1; then
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
      return 0
    fi
  done
  return 1
}

if ! find_conda; then
  echo "conda was not found. Install Miniconda/Anaconda first, or load it before running this script." >&2
  exit 2
fi

eval "$(conda shell.bash hook)"

conda_env_exists() {
  if [[ "${ENV_NAME}" == */* ]]; then
    [[ -d "${ENV_NAME}/conda-meta" ]]
  else
    conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"
  fi
}

create_conda_env() {
  if [[ "${ENV_NAME}" == */* ]]; then
    mkdir -p "$(dirname "${ENV_NAME}")"
    conda create -y -p "${ENV_NAME}" "python=${PYTHON_VERSION}"
  else
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}"
  fi
}

if ! conda_env_exists; then
  create_conda_env
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip wheel setuptools packaging ninja

if [[ "${INSTALL_TORCH}" == "1" ]]; then
  python -m pip install torch torchvision torchaudio --index-url "${PYTORCH_INDEX_URL}"
fi

python -m pip install -e .
python -m pip install -r requirements.txt
python -m pip install -r third_party/verl/requirements.txt
python -m pip install huggingface_hub

if [[ "${INSTALL_VLLM}" == "1" ]]; then
  python -m pip install "${VLLM_SPEC}"
fi

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  python -m pip install flash-attn --no-build-isolation
fi

python - <<'PY'
import importlib
import sys

checks = ["yaml", "pandas", "pyarrow", "torch", "ray", "tensordict", "transformers"]
for name in checks:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"{name}: {version}")

try:
    import vllm
    print(f"vllm: {getattr(vllm, '__version__', 'unknown')}")
except Exception as exc:
    print(f"vllm import skipped/failed: {exc}")

print(f"python: {sys.version}")
PY

echo "Conda environment ready: ${ENV_NAME}"
