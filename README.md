# GRPO M2RL IF/Science Training

This repository is now focused on M2RL-style GRPO for Qwen 4B non-thinking instruction following and science QA.

The active recipes are:

- `m2rl_if`: IFBench-style instruction-following reward.
- `m2rl_science`: GPQA-style science multiple-choice reward.
- `m2rl_if_science_mix`: 50/50 IF + Science mixed-domain GRPO.

Legacy ToolRL and General-Reasoner `grpo/` files were archived under the path recorded in `temp/latest_grpo_legacy_backup.txt`.

## Layout

```text
GRPO/
  grpo/
    configs/              # M2RL IF/Science GRPO configs
    data/m2rl.py          # M2RL/Nemotron-style data converter and validator
    rewards/m2rl.py       # IFBench and GPQA reward functions
  grpo_training/          # standalone launcher and data preparation CLI
  mopd_verl/              # compatibility layer required by the patched verl runtime
  scripts/                # launch scripts
  tests/                  # lightweight local tests
  third_party/verl/       # patched verl package used by these recipes
```

## Setup

```bash
cd /Users/linghuazhang/Desktop/Project/GRPO
uv venv
uv pip install -e .
uv pip install -r third_party/verl/requirements.txt
export PYTHONPATH="$PWD:$PWD/third_party/verl:${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS=0
```

For real training, install the GPU runtime required by the vendored verl stack, including compatible `torch`, `vllm`, `ray`, and `tensordict`.

## Data

M2RL does not ship ready-to-run public train parquet files in this repo. Prepare local files into verl schema:

```bash
python3 -m grpo_training.prepare_data prepare-m2rl \
  --input /path/to/if_train.parquet \
  --output data/M2RL/if/train.parquet \
  --rm-type ifbench \
  --domain if \
  --split train

python3 -m grpo_training.prepare_data prepare-m2rl \
  --input /path/to/science_train.parquet \
  --output data/M2RL/science/train.parquet \
  --rm-type gpqa \
  --domain science \
  --split train
```

Accepted input formats are `.parquet`, `.json`, and `.jsonl`.

IF rows need `instruction_id_list`, `kwargs`, and `prompt_text` in `metadata`, `extra_info`, or top-level columns. The IF reward calls official `allenai/IFBench` strict evaluation, so set one of:

```bash
export IFBENCH_REPO=/path/to/IFBench
# or allow startup to clone it into ./IFBench
export M2RL_ALLOW_IFBENCH_AUTO_CLONE=1
```

Science rows need a `correct_letter` or a label/answer plus choices. The GPQA scorer extracts final answer letters from non-thinking outputs and does not require `</think>`.

Default validation paths expected by the configs:

```text
eval/domains/ifbench/data/IFBench_test.parquet
eval/domains/science/data/gpqa.parquet
```

## Context Length

The M2RL configs match the released Qwen3-4B scripts:

- prompt length: `2048`
- response length: `32768`
- rollout max model length: `34816`
- samples per prompt: `16`
- train batch size: `2048`
- tensor parallel size: `2`
- non-thinking mode: `data.enable_thinking=false`

The model path defaults to `Qwen/Qwen3-4B`; override `model.student_path`, `model.student_base_path`, and `model.primary_teacher_path` when using a local Qwen 4B non-thinking checkpoint.

## Dry Run

```bash
scripts/run_m2rl_if_grpo.sh --dry-run
scripts/run_m2rl_science_grpo.sh --dry-run
scripts/run_m2rl_if_science_grpo.sh --dry-run
scripts/run_grpo.sh grpo/configs/m2rl_if_smoke.yaml --dry-run
```

Useful override:

```bash
scripts/run_m2rl_if_science_grpo.sh --dry-run -- \
  actor_rollout_ref.model.path=/models/qwen4b-non-thinking \
  actor_rollout_ref.model.base_model_path=/models/qwen4b-non-thinking
```

## Train

```bash
scripts/run_m2rl_if_grpo.sh
scripts/run_m2rl_science_grpo.sh
scripts/run_m2rl_if_science_grpo.sh
```

Set `WANDB_MODE=disabled` for local smoke tests that should not log to W&B.

## Remote Setup

Fill the remote SSH settings first:

```bash
cp scripts/ssh.example.sh scripts/ssh.sh
$EDITOR scripts/ssh.sh
```

`scripts/ssh.sh` is gitignored because it may contain server addresses or private key paths.

Sync code to the remote machine:

```bash
scripts/sync_remote.sh
```

Sync and install a conda environment remotely:

```bash
RUN_REMOTE_INSTALL=1 scripts/sync_remote.sh
```

The conda installer is:

```bash
scripts/install_conda_env.sh
```

Useful environment overrides:

```bash
REMOTE_CONDA_ENV=grpo-rl
REMOTE_PYTHON_VERSION=3.10
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
VLLM_SPEC=vllm==0.8.4
INSTALL_FLASH_ATTN=0
```

Download models and stage datasets on the remote machine:

```bash
scripts/download_assets.sh
```

By default it downloads `Qwen/Qwen3-4B` and clones IFBench. To prepare local dataset files into verl parquet schema, set source paths:

```bash
IF_TRAIN_SOURCE=/path/to/raw_if_train.parquet \
IF_VAL_SOURCE=/path/to/raw_if_val.parquet \
SCIENCE_TRAIN_SOURCE=/path/to/raw_science_train.parquet \
SCIENCE_VAL_SOURCE=/path/to/raw_science_val.parquet \
scripts/download_assets.sh
```

Run a remote smoke dry-run through SSH:

```bash
scripts/remote_smoke.sh
```

Run the actual one-step remote smoke by setting:

```bash
REMOTE_SMOKE_DRY_RUN=0 scripts/remote_smoke.sh
```
