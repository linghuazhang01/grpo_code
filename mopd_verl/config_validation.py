"""Cross-field validation for typed ``mopd_verl`` launcher settings."""

from __future__ import annotations

from typing import Any


def validate_mopd_config(config: Any) -> None:
    """Reject internally inconsistent settings before allocating remote GPUs."""

    data = config.data
    actor = config.actor
    rollout = config.rollout
    trainer = config.trainer
    world_size = trainer.n_gpus_per_node * trainer.nnodes
    sequence_length = data.max_prompt_length + data.max_response_length

    positive_values = {
        "data.train_batch_size": data.train_batch_size,
        "data.max_prompt_length": data.max_prompt_length,
        "data.max_response_length": data.max_response_length,
        "actor.ppo_mini_batch_size": actor.ppo_mini_batch_size,
        "actor.ppo_micro_batch_size_per_gpu": actor.ppo_micro_batch_size_per_gpu,
        "actor.ppo_max_token_len_per_gpu": actor.ppo_max_token_len_per_gpu,
        "rollout.tensor_model_parallel_size": rollout.tensor_model_parallel_size,
        "rollout.n": rollout.n,
        "rollout.max_num_batched_tokens": rollout.max_num_batched_tokens,
        "rollout.max_num_seqs": rollout.max_num_seqs,
        "trainer.n_gpus_per_node": trainer.n_gpus_per_node,
        "trainer.nnodes": trainer.nnodes,
    }
    for key, value in positive_values.items():
        if value <= 0:
            raise ValueError(f"Expected '{key}' to be positive, got {value}.")

    if data.val_batch_size is not None and data.val_batch_size <= 0:
        raise ValueError("Expected 'data.val_batch_size' to be positive or null.")
    if actor.model_dtype not in {"fp32", "float32"}:
        raise ValueError(
            "Expected 'actor.model_dtype' to be fp32/float32. The vendored verl FSDP "
            "worker must initialize actor optimizer states in FP32."
        )
    if world_size % rollout.tensor_model_parallel_size != 0:
        raise ValueError(
            "Total GPUs must be divisible by rollout.tensor_model_parallel_size: "
            f"world_size={world_size}, tensor_model_parallel_size={rollout.tensor_model_parallel_size}."
        )
    actor_batch_size = actor.ppo_mini_batch_size * rollout.n
    train_batch_size = data.train_batch_size * rollout.n
    if train_batch_size % world_size != 0:
        raise ValueError(
            "data.train_batch_size * rollout.n must be divisible by total GPUs: "
            f"{data.train_batch_size} * {rollout.n} is not divisible by {world_size}."
        )
    if actor_batch_size % world_size != 0:
        raise ValueError(
            "actor.ppo_mini_batch_size * rollout.n must be divisible by total GPUs: "
            f"{actor.ppo_mini_batch_size} * {rollout.n} is not divisible by {world_size}."
        )
    if actor.ppo_max_token_len_per_gpu < sequence_length:
        raise ValueError(
            "actor.ppo_max_token_len_per_gpu must cover max prompt + response length: "
            f"{actor.ppo_max_token_len_per_gpu} < {sequence_length}."
        )
    if rollout.max_model_len is not None and rollout.max_model_len < sequence_length:
        raise ValueError(
            "rollout.max_model_len must cover max prompt + response length: "
            f"{rollout.max_model_len} < {sequence_length}."
        )
    if (
        rollout.enable_chunked_prefill
        and rollout.max_model_len is not None
        and rollout.max_num_batched_tokens < rollout.max_model_len
    ):
        raise ValueError(
            "rollout.max_num_batched_tokens must be >= rollout.max_model_len when "
            "chunked prefill is enabled."
        )
    for key, value in (
        ("trainer.max_actor_ckpt_to_keep", trainer.max_actor_ckpt_to_keep),
        ("trainer.max_critic_ckpt_to_keep", trainer.max_critic_ckpt_to_keep),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"Expected '{key}' to be positive or null.")
    if trainer.total_training_steps is not None and trainer.total_training_steps <= 0:
        raise ValueError("Expected 'trainer.total_training_steps' to be positive or null.")
    if trainer.resume_mode not in {"auto", "disable", "resume_path"}:
        raise ValueError(
            "Expected 'trainer.resume_mode' to be 'auto', 'disable', or 'resume_path'."
        )
