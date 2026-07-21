# M2RL GRPO Package

Active files:

- `data/m2rl.py`: validates and converts IFBench/Science data into verl parquet schema.
- `rewards/m2rl.py`: dispatches IFBench strict reward and GPQA multiple-choice reward.
- `configs/m2rl_if.yaml`: IF-only GRPO.
- `configs/m2rl_science.yaml`: Science-only GRPO.
- `configs/m2rl_science_smoke_2gpu.yaml`: two-step Science smoke run on 2 GPUs.
- `configs/m2rl_science_{6,8}gpu_141gb.yaml`: standalone Science production profiles.
- `configs/m2rl_if_{6,8}gpu_141gb.yaml`: standalone IF production profiles.
- `configs/m2rl_if_science_mix.yaml`: mixed IF + Science GRPO.

The Science and IF production configs use `2048 + 16384 = 18432` tokens,
128 prompts per update, and rollout `n=16` (2048 trajectories). The separate
2-GPU smoke config uses a 512-token response and runs only two training steps.
