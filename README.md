# GRPO：M2RL IF / Science Training

本仓库基于 vendored `verl`，用于在 Qwen3-4B 上运行 M2RL-style GRPO。当前提供三套 recipe：

- `m2rl_science`：Science multiple-choice QA，使用 GPQA-style 字母答案 reward。
- `m2rl_if`：Instruction Following，使用 IFBench strict reward。
- `m2rl_if_science_mix`：IF 与 Science 的 50/50 mixed-domain GRPO。

> [!IMPORTANT]
> 本项目中 `science` 表示 text-only Science MCQA 数据域。当前已验证的训练集来自 NVIDIA Nemotron RL blend 的 `knowledge-mcqa` 子集，不是[官方 multimodal ScienceQA](https://scienceqa.github.io/)，也不是 GPQA 官方训练集。若使用官方 ScienceQA，需要先把问题、选项以及必要的图像信息转成纯文本 prompt，并保留 `correct_letter`；当前训练 pipeline 不直接读取图片。

## 目录结构

```text
GRPO/
├── grpo/
│   ├── configs/                 # IF / Science / mixed GRPO 配置
│   ├── data/m2rl.py             # 数据转换与 schema 校验
│   └── rewards/m2rl.py          # IFBench 与 GPQA-style reward
├── mopd_verl/                   # 唯一训练 launcher、配置解析与数据准备 CLI
├── scripts/                     # 环境、数据、训练和远端脚本
├── tests/                       # 本地单元测试
└── third_party/verl/            # 本项目实际使用的 vendored verl
```

## 1. 环境说明

### 1.1 已验证环境

远端机器已经有可直接复用的 Conda 环境：

```text
Conda environment: mopd-verl
Python module:      mopd_verl
Python:             3.10.20
PyTorch:            2.6.0+cu124
CUDA runtime:       12.4
vLLM:               0.8.5.post1
Ray:                2.47.1
Transformers:       4.51.3
TensorDict:         0.6.2
FlashAttention:     2.7.4.post1
GPU:                2 × NVIDIA GeForce RTX 3090（每张 48 GiB）
```

环境名是 `mopd-verl`（hyphen），代码包名是 `mopd_verl`（underscore），二者不要混用。仓库已移除重复的 `grpo_training` package；训练 launcher、配置解析、W&B 环境加载和数据准备现在全部以 `mopd_verl` 为唯一入口。该环境已通过 PyTorch CUDA、FlashAttention、Qwen3-4B load 和本仓库 tests 验证。

登录机器后，推荐直接复用：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mopd-verl

cd /root/autodl-tmp/GRPO
export PYTHONPATH="$PWD:$PWD/third_party/verl:${PYTHONPATH:-}"
export PYTHONINTMAXSTRDIGITS=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
```

检查环境：

```bash
python - <<'PY'
import sys

import ray
import tensordict
import torch
import transformers
import vllm
import flash_attn

print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
print("vllm:", vllm.__version__)
print("ray:", ray.__version__)
print("transformers:", transformers.__version__)
print("tensordict:", tensordict.__version__)
print("flash_attn:", flash_attn.__version__)
PY

python -m pytest -q
```

第一次访问 CUDA 时可能需要约 30 秒初始化，不应仅因短暂无输出就中止检查。

### 1.2 新建环境（仅在无法复用时）

优先复用现有 `mopd-verl`。如果必须新建，可使用仓库脚本：

```bash
REMOTE_CONDA_ENV=mopd-verl \
REMOTE_PYTHON_VERSION=3.10 \
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 \
VLLM_SPEC=vllm==0.8.5.post1 \
INSTALL_FLASH_ATTN=1 \
bash scripts/install_conda_env.sh
```

该脚本不会锁定所有 transitive dependencies；新建后必须重新运行上面的版本和 CUDA 检查。开发机只做数据处理和 unit tests 时，也可使用：

```bash
uv venv
uv pip install -e .
uv pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/third_party/verl:${PYTHONPATH:-}"
python -m pytest -q
```

### 1.3 W&B 配置（使用 `.env.local`）

本项目已支持 W&B：production recipes 默认使用
`trainer.logger=["console","wandb"]`，vendored `verl` 的依赖中也包含 `wandb`。
Launcher 会在启动训练子进程时自动读取项目根目录的 `.env.local`，无需手动执行
`wandb login`，也不会把 API key 拼进训练命令或输出到日志。

先从安全模板创建本地配置：

```bash
cd /root/autodl-tmp/GRPO
cp .env.local.example .env.local
chmod 600 .env.local
$EDITOR .env.local
```

填写以下内容：

```bash
export WANDB_API_KEY=your_wandb_api_key_here
export WANDB_MODE=online

# 可选：使用 team/entity 时取消注释。
# export WANDB_ENTITY=your_team_or_username
```

`.env.local` 已被 `.gitignore` 排除，不能提交或通过同步脚本公开；仓库只保留不含
secret 的 `.env.local.example`。变量优先级为：当前 shell 环境 > `.env.local` > YAML
中的 `runtime.wandb_mode`。因此可对单次任务临时关闭联网记录：

```bash
WANDB_MODE=disabled bash scripts/run_m2rl_science_grpo.sh -- \
  'trainer.logger=["console"]'
```

所有正式 recipe 都显式配置了 `runtime.env_file: .env.local`。若不希望加载任何 env
文件，可在自定义 YAML 中设为 `runtime.env_file: null`。

141 GB profiles 会在 checkpoint 目录保存一个权限为 `0600` 的 `.wandb_run_id`，并
设置 `WANDB_RESUME=allow`。因此同一 checkpoint 目录断点续训时会继续写入同一个
W&B run，而不是创建同名的新 run。新实验应使用新的 `GRPO_CHECKPOINT_DIR`；如需
手动接管 run，也可在 `.env.local` 中设置 `WANDB_RUN_ID` 和 `WANDB_RESUME`。

## 2. 模型准备

当前验证使用 Qwen3-4B。可直接引用已有模型目录，也可下载到本地：

```bash
hf download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
export GRPO_MODEL_PATH="$PWD/models/Qwen3-4B"
```

远端已验证的模型路径是 `/root/OPD/models/Qwen3-4B`：

```bash
export GRPO_MODEL_PATH=/root/OPD/models/Qwen3-4B
test -f "$GRPO_MODEL_PATH/config.json"
```

这里的 `Qwen3-4B` 是约 4.02B 参数的 post-trained Qwen3 checkpoint；模型卡中的
`base_model: Qwen/Qwen3-4B-Base` 表示它由 Base checkpoint 继续训练而来，不代表当前
加载的是 `Qwen3-4B-Base`。non-thinking 是 chat-template 运行模式，由 recipe 的
`enable_thinking: false` 控制，不是另一套权重。

## 3. 数据准备

训练文件最终必须是 verl parquet schema，至少包含：

```text
data_source, prompt, ability, reward_model, extra_info
```

Science 样本必须能解析出 choices 和正确答案字母；IF 样本必须包含 IFBench evaluator 所需的 constraint metadata。

### 3.1 推荐：准备 Nemotron Science / IF 数据

下载 [NVIDIA Nemotron-3-Nano-RL-Training-Blend](https://huggingface.co/datasets/nvidia/Nemotron-3-Nano-RL-Training-Blend)：

```bash
mkdir -p data/raw/nemotron-3-nano-rl-training-blend
hf download nvidia/Nemotron-3-Nano-RL-Training-Blend \
  train.jsonl README.md create_nanov3_jsonl.py \
  --repo-type dataset \
  --local-dir data/raw/nemotron-3-nano-rl-training-blend
```

完整 `train.jsonl` 约 6.9 GB，准备前请先确认磁盘空间。转换为当前 recipe 所需 parquet：

```bash
python scripts/prepare_nemotron_rl_data.py --write-raw-splits
```

默认输出：

```text
data/nemotron_rl/manifest.json
data/nemotron_rl/splits/*.jsonl
data/M2RL/if/train.parquet
data/M2RL/science/train.parquet
```

已下载的 raw manifest 共 93,244 条，其中 Science MCQA 19,670 条、IF 16,575 条。
其中 4 条 Science raw records 的 `expected_answer` 不在其 option labels 中；当前
converter 会跳过这些不可评分记录，并把详情写入 manifest 的 `invalid_rows`，所以
该版本应产出 19,666 条可用 Science records。旧版本生成的 parquet 必须重新转换。
实际训练前必须再次校验当前下载版本，不要只依赖这些历史计数：

```bash
python -m mopd_verl.prepare_data inspect data/M2RL/science/train.parquet
python -m mopd_verl.prepare_data inspect data/M2RL/if/train.parquet
python -m grpo.data.m2rl validate \
  --input data/M2RL/science/train.parquet --rm-type gpqa
python -m grpo.data.m2rl validate \
  --input data/M2RL/if/train.parquet --rm-type ifbench
python -m json.tool data/nemotron_rl/manifest.json | less
```

两个 validator 都返回 code 0，且 `invalid_rows`、`sample_id_invalid_rows` 均为空，
才表示 metadata 与 reward label 基础检查通过。它不会替代内容级去重或 label 人工审查。

### 3.2 转换自定义 Science 数据

输入格式支持 `.parquet`、`.json` 和 `.jsonl`。常见字段如下：

- Prompt：`prompt`、`messages`、`question` 或 `input`。
- Label：`correct_letter`、`label`、`answer`、`ground_truth` 或 `target`。
- Choices：推荐在 metadata 或顶层提供 `choices`，并提供 `valid_letters`。
- 可选标识：`record_id`、`sample_id`。

转换命令：

```bash
python -m mopd_verl.prepare_data prepare-m2rl \
  --input /path/to/science_train.jsonl \
  --output data/M2RL/science/train.parquet \
  --rm-type gpqa \
  --domain science \
  --split train

python -m mopd_verl.prepare_data inspect \
  data/M2RL/science/train.parquet
```

先用小样本检查 schema 和 reward：

```bash
python -m mopd_verl.prepare_data prepare-m2rl \
  --input /path/to/science_train.jsonl \
  --output data/M2RL/science/train_smoke.parquet \
  --rm-type gpqa \
  --domain science \
  --max-samples 32
```

### 3.3 转换自定义 IF 数据

IF row 的 top-level、`metadata` 或 `extra_info` 中需要包含 `instruction_id_list`、`kwargs` 和 `prompt_text`：

```bash
python -m mopd_verl.prepare_data prepare-m2rl \
  --input /path/to/if_train.parquet \
  --output data/M2RL/if/train.parquet \
  --rm-type ifbench \
  --domain if \
  --split train
```

IF reward 使用官方 IFBench strict evaluator。请设置已有 checkout，或显式允许首次运行时 clone：

```bash
export IFBENCH_REPO=/path/to/IFBench
# 或：export M2RL_ALLOW_IFBENCH_AUTO_CLONE=1
```

### 3.4 Validation 数据

默认配置期望：

```text
eval/domains/science/data/gpqa.parquet
eval/domains/ifbench/data/IFBench_test.parquet
```

也可在启动命令中通过 `data.val_files` 覆盖。请确保 validation 与 train 没有 prompt、题目或选项级的数据泄漏。

## 4. 启动训练

所有 wrapper 最终都调用 `python -m mopd_verl.launch`，不再存在第二套训练
launcher。

### 4.1 先做 dry run

Dry run 只构造并打印最终的 verl/Hydra 命令，不分配 GPU：

```bash
scripts/run_m2rl_science_grpo.sh --dry-run
scripts/run_m2rl_if_grpo.sh --dry-run
scripts/run_m2rl_if_science_grpo.sh --dry-run
```

Hydra override 必须放到 `--` 之后：

```bash
scripts/run_m2rl_science_grpo.sh --dry-run -- \
  actor_rollout_ref.model.path="$GRPO_MODEL_PATH" \
  actor_rollout_ref.model.base_model_path="$GRPO_MODEL_PATH" \
  actor_rollout_ref.ref.model.path="$GRPO_MODEL_PATH"
```

### 4.2 6/8 × 141 GB GPU 正式训练

Science 与 IF 都提供了独立的 6-GPU、8-GPU 启动脚本：

| Domain | GPUs | Global prompt batch | Validation batch | Rollouts/prompt | Trajectories/GPU/step |
|---|---:|---:|---:|---:|---:|
| Science | 6 | 126 | 48 | 16 | 336 |
| Science | 8 | 128 | 64 | 16 | 256 |
| IF | 6 | 126 | 48 | 16 | 336 |
| IF | 8 | 128 | 64 | 16 | 256 |

M2RL 论文中的 batch 口径是 128 个 prompts、每个 prompt 生成 16 条 rollout，即每次
update 共 2048 条 trajectories。verl 的 `data.train_batch_size` 表示 prompt 数，不能把
2048 直接填入该字段，否则会生成 32768 条 trajectories。8 GPU profile 精确复现
`128 × 16 = 2048`；6 GPU 为满足 batch 可被 data-parallel world size 整除，使用最接近
论文的 `126 × 16 = 2016`。`max_num_seqs=32` 只限制 vLLM 同时驻留的 sequence 数，
不改变 logical rollout batch。共同参数为：

- `max_prompt_length=2048`
- `max_response_length=16384`
- `max_model_len=18432`
- `rollout.n=16`、`tensor_model_parallel_size=1`
- actor 以 FP32 初始化，避免 Adam optimizer state 被错误创建为 BF16；forward/backward
  仍使用 verl 的 BF16 mixed precision
- 每 10 steps 保存一次，最多保留 5 个完整、可恢复的 checkpoint directories
- `total_training_steps=1200`，validation 每 50 steps 执行一次
- 默认 `resume_mode=auto`，相同 checkpoint 目录会自动续训

正式脚本会在分配 GPU 前检查 visible GPU 数量、model/data 文件、M2RL schema；IF
还会检查 IFBench evaluator 是否可 import。完成 step 1200 后再次执行同一命令会直接
退出，不会错误地继续训练 step 1201。只有在单纯检查 Hydra compose 且明确没有准备
assets 时，才可临时设置 `GRPO_SKIP_PREFLIGHT=1`。

先确认四条命令：

```bash
scripts/run_m2rl_science_6gpu_141gb.sh --dry-run
scripts/run_m2rl_science_8gpu_141gb.sh --dry-run
scripts/run_m2rl_if_6gpu_141gb.sh --dry-run
scripts/run_m2rl_if_8gpu_141gb.sh --dry-run
```

远端正式启动前设置模型与数据路径：

```bash
export GRPO_MODEL_PATH=/root/OPD/models/Qwen3-4B
export SCIENCE_TRAIN_FILE=/path/to/science/train.parquet
export SCIENCE_VAL_FILE=/path/to/science/validation.parquet
export IF_TRAIN_FILE=/path/to/if/train.parquet
export IF_VAL_FILE=/path/to/if/validation.parquet
```

选择一条启动；脚本会自动设置前 6 或前 8 张 GPU，也可以提前设置
`CUDA_VISIBLE_DEVICES` 覆盖，但其数量必须与所选 6/8-GPU profile 一致：

```bash
mkdir -p logs checkpoints

scripts/run_m2rl_science_6gpu_141gb.sh \
  2>&1 | tee logs/science-6gpu-141gb-16k.log

scripts/run_m2rl_science_8gpu_141gb.sh \
  2>&1 | tee logs/science-8gpu-141gb-16k.log

scripts/run_m2rl_if_6gpu_141gb.sh \
  2>&1 | tee logs/if-6gpu-141gb-16k.log

scripts/run_m2rl_if_8gpu_141gb.sh \
  2>&1 | tee logs/if-8gpu-141gb-16k.log
```

额外 Hydra 参数仍放在 `--` 后，但 141 GB profile 的 GPU 数、data/model path、batch、
length、checkpoint 与 step 上限属于受保护参数，不能被尾部 override 绕过。路径请使用
对应环境变量；例如启动一个新的 checkpoint/W&B run：

```bash
GRPO_CHECKPOINT_DIR=checkpoints/science-fresh-run \
scripts/run_m2rl_science_8gpu_141gb.sh -- trainer.resume_mode=disable
```

### 4.3 两张 GPU 的 smoke run

[`grpo/configs/m2rl_science_smoke_2gpu.yaml`](grpo/configs/m2rl_science_smoke_2gpu.yaml)
是独立的 2 GPU Science smoke 配置，不会继承正式训练的 16K response 和 2048 条
trajectories。它固定使用 4 prompts × 4 rollouts、512-token response、2 steps，关闭
W&B 和 checkpoint，但会执行 rollout、Science reward、actor update 和 validation。
该配置已按 2×48 GiB RTX 3090 的显存规模设置，在 2×141 GB GPU 上也可直接运行。

先检查最终命令：

```bash
scripts/run_m2rl_science_2gpu_smoke.sh --dry-run
```

准备路径并执行实际 smoke：

```bash
mkdir -p logs checkpoints
export GRPO_MODEL_PATH=/root/OPD/models/Qwen3-4B
export SCIENCE_TRAIN_FILE=/path/to/science/train.parquet
export SCIENCE_VAL_FILE=/path/to/science/validation.parquet

CUDA_VISIBLE_DEVICES=0,1 PYTHONUNBUFFERED=1 \
scripts/run_m2rl_science_2gpu_smoke.sh \
  2>&1 | tee logs/science-2gpu-smoke.log
```

> [!WARNING]
> Launcher 默认强制设置 `attn_implementation=flash_attention_2`，因此 actor/reference 的核心 attention forward 使用 FlashAttention 2。`grpo/configs/m2rl_science.yaml` 仍保留 `actor_rollout_ref.model.use_remove_padding=False`：它只关闭已知有 parity 问题的 verl varlen/remove-padding 路径，不会把 Hugging Face attention backend 改回 `eager`。在修复并重新做 parity test 前不要启用 `remove_padding=True`。

Qwen3 的 non-thinking template 仍会在格式化后的 prompt 中生成空的
`<think>\n\n</think>` 占位符；标签之间没有 reasoning token，这是
`enable_thinking=False` 的预期行为。若原始题目本身要求 “Think step by step”，模型仍可能
在普通 response 文本里给出推导，这不表示 thinking mode 被重新开启。

### 4.4 调整训练规模

Pilot 指标和生成样本正常后，再逐项增加：

1. `trainer.total_training_steps` 和 checkpoint 频率。
2. `data.train_batch_size` / `ppo_mini_batch_size`。
3. `actor_rollout_ref.rollout.n`（每个 prompt 的采样数）。
4. `data.max_response_length` 与 `rollout.max_model_len`。
5. W&B logging：按 1.3 节创建 `.env.local`；recipe 已默认启用
   `WANDB_MODE=online` 和 `trainer.logger=["console","wandb"]`。

Science/IF 的默认 8-GPU recipe 已与 141 GB profile 对齐：prompt batch 128、response
length 16384、每题采样 16 次、tensor parallel 1。不要通过提高
`ppo_micro_batch_size_per_gpu` 来追求 logical batch；它保持为 1，并由 dynamic batch 和
gradient accumulation 完成一次 2048-trajectory update。

IF 和 mixed-domain 使用相应 wrapper：

```bash
scripts/run_m2rl_if_grpo.sh -- <Hydra overrides...>
scripts/run_m2rl_if_science_grpo.sh -- <Hydra overrides...>
```

## 5. 远端同步与后台运行

不要把 SSH password、token 或 private key 写入仓库。复制 gitignored 配置模板：

```bash
cp scripts/ssh.example.sh scripts/ssh.sh
$EDITOR scripts/ssh.sh
scripts/sync_remote.sh
```

若远端已经有 `mopd-verl`，无需重复安装。远端 smoke check：

```bash
REMOTE_CONDA_ENV=mopd-verl scripts/remote_smoke.sh
```

长任务推荐使用 `screen` 或 `tmux`，示例：

```bash
screen -S science-grpo
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mopd-verl
cd /root/autodl-tmp/GRPO
# 在此粘贴 4.2 的训练命令
```

退出但保留任务：`Ctrl-A D`；重新进入：

```bash
screen -r science-grpo
tail -f logs/science-grpo-pilot.log
nvidia-smi
```

## 6. 输出、验证与故障排查

- Logs：`logs/`
- Checkpoints：`checkpoints/`
- 实验分析：`plan/science-grpo-analysis/analysis-report.md`
- Tests：`python -m pytest -q`

常见问题：

- `ModuleNotFoundError: verl`：确认 `PYTHONPATH` 同时包含 repo root 和 `third_party/verl`。
- CUDA OOM：先降低 `rollout.n`、`max_response_length`、`max_num_seqs` 和 `gpu_memory_utilization`。
- response 大量被截断：在显存允许时同步提高 `data.max_response_length`、`rollout.max_model_len` 和 actor token limit。
- Actor/vLLM KL 异常大：先确认没有覆盖 `use_remove_padding=False`，再检查模型 path 和 rollout log-prob parity。
- Reward 长期全 0：抽查生成答案是否包含可解析的最终选项字母，并检查 `correct_letter` 与 `valid_letters`。
- Resume 到旧实验：使用新的 `trainer.default_local_dir`，或显式设置 `trainer.resume_mode=disable`。
- 数据预检报告 `correct_letter` 不在 `valid_letters`：重新运行 3.1 的 converter，
  不要使用旧版生成的 Science parquet。
- W&B 未记录：检查 `.env.local` 是否存在且 `WANDB_API_KEY` 非空，并确认没有在 shell 中设置 `WANDB_MODE=disabled`。
- 磁盘不足：Nemotron 原始数据、vLLM cache 和 checkpoints 都较大；4B actor 的模型、
  optimizer 与 scheduler state 会让单个 checkpoint 达到数十 GB，保留 5 份前应先用
  实际远端 `df -h` 与首个 checkpoint 大小估算容量。

## 7. 当前验证结论

当前 pipeline 已完成 2×RTX 3090、Qwen3-4B、Science MCQA 的两步端到端 smoke：
数据加载、vLLM rollout、reward、actor update 和 validation 均可运行。实际 worker 日志确认
加载的是 `/root/OPD/models/Qwen3-4B`、模型规模 4.02B、
`attn_implementation=flash_attention_2`、`enable_thinking=False`，同时保持
`use_remove_padding=False`。step 1/2 的 training reward 分别为 `0.1875`/`0.3125`，KL
分别为 `0.000612`/`0.000497`，actor/vLLM log-prob correlation 分别为
`0.998610`/`0.998554`；两步 response aborted ratio 和 clip ratio 均为 0，进程 exit code
为 0，未出现 OOM、NaN 或 traceback。

Transformers 会在 FSDP 包装前针对 FP32 actor 初始化打印一次 FlashAttention dtype warning；
这是 verl 为保证 Adam state 正确而采用的初始化方式，forward/backward 随后使用 BF16 mixed
precision。完整 smoke 已成功经过 rollout 和两次参数更新，因此该 warning 不是 backend
fallback 或训练失败。Smoke 只验证训练链路，不能据此宣称模型能力已稳定提升；正式结论仍
需要更长训练、多个 random seeds，以及独立 held-out evaluation。
