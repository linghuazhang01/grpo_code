# M2RL GRPO Package

Active files:

- `data/m2rl.py`: validates and converts IFBench/Science data into verl parquet schema.
- `rewards/m2rl.py`: dispatches IFBench strict reward and GPQA multiple-choice reward.
- `configs/m2rl_if.yaml`: IF-only GRPO.
- `configs/m2rl_science.yaml`: Science-only GRPO.
- `configs/m2rl_if_science_mix.yaml`: mixed IF + Science GRPO.

The active context setting is `2048 + 32768 = 34816` tokens with rollout `n=16`.

