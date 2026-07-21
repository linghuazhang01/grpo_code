"""Typed configuration for the mopd_verl GRPO launcher."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from mopd_verl.config_validation import validate_mopd_config

DEFAULT_PAPER_EVAL_DATASETS = [
    "aime24",
    "aime25",
    "hmmt25_feb",
    "hmmt25_nov",
    "humaneval_plus",
    "mbpp_plus",
    "lcb",
]


@dataclass(frozen=True)
class DataConfig:
    train_files: list[str]
    val_files: list[str]
    domain_train_files: dict[str, list[str]] = field(default_factory=dict)
    domain_sampling_weights: dict[str, float] = field(default_factory=dict)
    domain_sampling_replacement: bool = True
    train_batch_size: int = 1024
    val_batch_size: int | None = None
    max_prompt_length: int = 2048
    max_response_length: int = 16384
    filter_overlong_prompts: bool = True
    truncation: str = "error"
    shuffle: bool = True
    validation_shuffle: bool = False
    seed: int = 42
    return_raw_chat: bool = True
    enable_thinking: bool = False
    need_tools_kwargs: bool = False


@dataclass(frozen=True)
class ModelConfig:
    student_path: str
    student_base_path: str | None
    math_teacher_path: str
    code_teacher_path: str
    reasoning_teacher_path: str | None
    primary_teacher_path: str
    secondary_teacher_path: str | None
    teacher_model_device: str = "cpu"
    attn_implementation: str = "flash_attention_2"
    use_remove_padding: bool = False
    reference_param_offload: bool = True


@dataclass(frozen=True)
class ActorConfig:
    learning_rate: str = "1e-5"
    lr_warmup_steps_ratio: float = 0.0
    only_reverse_kl_advantages: bool = True
    lambda_vals: float = 1.25
    multi_teacher_distill: bool = True
    distill_loss_builder: str = "auto"
    distill_mode: str = "chosen_token_reverse_kl"
    topk_distill_enabled: bool = False
    topk_distill_kl_direction: str = "reverse"
    topk_distill_k: int = 8
    topk_distill_support_source: str = "teacher"
    topk_distill_tail_bucket: bool = True
    topk_distill_temperature: float = 1.0
    topk_distill_loss_weight: float = 1.0
    topk_distill_logprob_chunk_size: int = 16
    topk_distill_logprob_mode: str = "sparse"
    teacher_prefix_enabled: bool = False
    teacher_prefix_loss_region: str = "suffix_only"
    teacher_prefix_forward_kl_weight: float = 1.0
    ppo_mini_batch_size: int = 1024
    ppo_micro_batch_size_per_gpu: int = 1
    use_dynamic_bsz: bool = False
    use_kl_loss: bool = True
    kl_loss_coef: float = 0.0
    kl_loss_type: str = "low_var_kl"
    entropy_coeff: float = 0.0
    ppo_max_token_len_per_gpu: int = 32768
    gradient_checkpointing: bool = True
    param_offload: bool = False
    optimizer_offload: bool = False
    model_dtype: str = "fp32"
    fsdp_size: int | None = None


@dataclass(frozen=True)
class RolloutConfig:
    calculate_log_probs: bool = True
    log_prob_micro_batch_size_per_gpu: int = 4
    tensor_model_parallel_size: int = 4
    name: str = "vllm"
    mode: str = "sync"
    gpu_memory_utilization: float = 0.6
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False
    n: int = 1
    max_num_batched_tokens: int = 32768
    max_model_len: int | None = None
    max_num_seqs: int = 1024
    num_gpu_blocks_override: int | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    teacher_prefix_sampling_enabled: bool = False
    teacher_prefix_length: int = 1024
    teacher_prefix_dataset_key: str = "prefix"
    val_n: int = 1
    val_do_sample: bool = False
    val_temperature: float = 1.0
    val_top_p: float = 1.0
    seed: int = 42
    multi_turn_enable: bool = False
    multi_turn_tool_config_path: str | None = None
    multi_turn_max_assistant_turns: int | None = None
    multi_turn_max_user_turns: int | None = None
    multi_turn_max_parallel_calls: int = 1
    multi_turn_max_tool_response_length: int = 256
    multi_turn_tool_response_truncate_side: str = "middle"
    multi_turn_format: str = "hermes"
    multi_turn_tokenization_sanity_check_mode: str = "strict"


@dataclass(frozen=True)
class RolloutCorrectionConfig:
    rollout_is: str = "token"
    rollout_is_threshold: float = 5.0
    rollout_rs: str | None = "null"
    bypass_mode: bool = False


@dataclass(frozen=True)
class AuditConfig:
    enabled: bool = False
    output_dir: str = "mopd_audit"
    domains: list[str] = field(default_factory=lambda: ["math", "code"])
    tensorboard_prefix: str = "mopd"
    tensorboard_layout: str = "domain_category"
    tensorboard_prune_mode: str = "none"
    loss_variance_signal: str = "opd_loss_token"
    max_samples_per_domain: int | None = None
    high_variance_cv_threshold: float = 1.0
    log_sample_level: bool = True
    log_sample_level_freq_steps: int = 1
    log_validation_metrics: bool = True
    log_validation_metrics_freq_steps: int = 1
    tier2_window_size: int = 20
    calibration_bins: int = 10
    full_gradient_enabled: bool = False
    full_gradient_freq_steps: int = 1
    full_grad_training_parity_freq_steps: int = 1
    full_gradient_train_max_samples_per_domain: int | None = None
    full_gradient_micro_batch_size_per_gpu: int = 1
    full_gradient_storage_dtype: str = "float32"
    execution_timing: str = "pre_update"
    full_gradient_direct_recompute_enabled: bool = True
    sequence_masked_target_enabled: bool = False
    sequence_masked_target_use_as_primary: bool = False
    sample_gradient_enabled: bool = False
    sample_gradient_freq_steps: int = 1
    sample_gradient_norm_enabled: bool = True
    sample_gradient_cos_enabled: bool = False
    sample_gradient_cos_freq_steps: int = 1
    sample_gradient_backward_recompute_enabled: bool = True
    sample_gradient_backward_sync_enabled: bool = True
    sample_gradient_log_sample_level: bool = True
    sample_gradient_log_sample_level_freq_steps: int = 1
    full_gradient_offload_domain_gradients: bool = True
    token_gap_enabled: bool = True
    token_gap_freq_steps: int = 1
    token_gap_vocab_vector_enabled: bool = False
    token_gap_vocab_vector_freq_steps: int = 1
    token_gap_vocab_size: int | None = None
    entropy_enabled: bool = True
    entropy_freq_steps: int = 1
    entropy_vocab_vector_enabled: bool = False
    entropy_vocab_vector_freq_steps: int = 1
    token_conflict_enabled: bool = True
    token_conflict_freq_steps: int = 1
    token_conflict_top_k: int | None = None
    token_gradient_enabled: bool = False
    token_gradient_freq_steps: int = 10
    token_gradient_gap_selection_enabled: bool = True
    token_gradient_gap_abs_selection_enabled: bool = True
    token_gradient_loss_abs_selection_enabled: bool = True
    token_gradient_top_k: int = 100
    token_gradient_top_p: float = 0.10
    token_gradient_strict_grad_restore: bool = False
    token_gradient_backward_recompute_enabled: bool = True
    token_gradient_backward_sync_enabled: bool = True


@dataclass(frozen=True)
class PaperEvalConfig:
    enabled: bool = False
    script_path: str | None = None
    model_path: str | None = None
    output_dir: str = "paper_eval"
    datasets: list[str] = field(default_factory=lambda: list(DEFAULT_PAPER_EVAL_DATASETS))
    run_on_initial_validation: bool = True
    evaluate_current_checkpoint: bool = True
    fail_on_error: bool = False
    timeout_seconds: int = 0


@dataclass(frozen=True)
class TrainerConfig:
    project_name: str = "on-policy-distillation"
    experiment_name: str = "Qwen3-4B-Non-Thinking-Multi-Teacher-Distill-ExOPD"
    logger: str = '["console","wandb"]'
    n_gpus_per_node: int = 8
    nnodes: int = 1
    save_freq: int = 50
    default_local_dir: str = "checkpoints/Qwen3-4B-Non-Thinking-Multi-Teacher-Distill-ExOPD"
    test_freq: int = 10
    total_epochs: int = 3
    total_training_steps: int | None = None
    max_actor_ckpt_to_keep: int | None = None
    max_critic_ckpt_to_keep: int | None = None
    resume_mode: str = "auto"
    critic_warmup: int = 0
    val_before_train: bool = True
    log_val_generations: int = 10


@dataclass(frozen=True)
class RayInitConfig:
    include_dashboard: bool | None = None
    num_cpus: int | None = None


@dataclass(frozen=True)
class RayKwargsConfig:
    ray_init: RayInitConfig = field(default_factory=RayInitConfig)


@dataclass(frozen=True)
class RuntimeConfig:
    python_bin: str = "python3"
    verl_module: str = "verl.trainer.main_ppo"
    wandb_mode: str = "online"
    wandb_entity: str | None = None
    env_file: str | None = ".env.local"
    used_model: str = "no_api"


@dataclass(frozen=True)
class MOPDConfig:
    data: DataConfig
    model: ModelConfig
    actor: ActorConfig = field(default_factory=ActorConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    rollout_correction: RolloutCorrectionConfig = field(default_factory=RolloutCorrectionConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    paper_eval: PaperEvalConfig = field(default_factory=PaperEvalConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    ray_kwargs: RayKwargsConfig = field(default_factory=RayKwargsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    extra_overrides: list[str] = field(default_factory=list)


def _expect_mapping(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping.")
    return value


def _string_list(value: Any, key: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise ValueError(f"Expected '{key}' to be a string or a list of strings.")


def _float_mapping(value: Any, key: str) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping.")
    output: dict[str, float] = {}
    for item_key, item_value in value.items():
        numeric = float(item_value)
        if numeric <= 0:
            raise ValueError(f"Expected '{key}.{item_key}' to be positive.")
        output[str(item_key)] = numeric
    return output


def _string_list_mapping(value: Any, key: str) -> dict[str, list[str]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{key}' to be a mapping.")
    output: dict[str, list[str]] = {}
    for item_key, item_value in value.items():
        output[str(item_key)] = _string_list(item_value, f"{key}.{item_key}")
    return output


def _optional_string(value: Any, key: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ValueError(f"Expected '{key}' to be a string or null.")


def _same_model_path(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    return os.path.normcase(os.path.normpath(str(left))) == os.path.normcase(os.path.normpath(str(right)))


def load_config(path: str | Path) -> MOPDConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _expect_mapping(raw, "root")

    data_raw = _expect_mapping(root.get("data", {}), "data")
    model_raw = _expect_mapping(root.get("model", {}), "model")
    paper_eval_raw = _expect_mapping(root.get("paper_eval", {}), "paper_eval")
    domain_train_files = _string_list_mapping(data_raw.get("domain_train_files"), "data.domain_train_files")
    train_files = (
        _string_list(data_raw.get("train_files"), "data.train_files")
        if data_raw.get("train_files") is not None
        else [file_path for files in domain_train_files.values() for file_path in files]
    )
    if not train_files:
        raise ValueError("Expected 'data.train_files' or 'data.domain_train_files' to contain at least one file.")

    data = DataConfig(
        train_files=train_files,
        val_files=_string_list(data_raw.get("val_files"), "data.val_files"),
        domain_train_files=domain_train_files,
        domain_sampling_weights=_float_mapping(
            data_raw.get("domain_sampling_weights"), "data.domain_sampling_weights"
        ),
        domain_sampling_replacement=bool(
            data_raw.get("domain_sampling_replacement", DataConfig.domain_sampling_replacement)
        ),
        train_batch_size=int(data_raw.get("train_batch_size", DataConfig.train_batch_size)),
        val_batch_size=(
            None if data_raw.get("val_batch_size") is None else int(data_raw["val_batch_size"])
        ),
        max_prompt_length=int(data_raw.get("max_prompt_length", DataConfig.max_prompt_length)),
        max_response_length=int(data_raw.get("max_response_length", DataConfig.max_response_length)),
        filter_overlong_prompts=bool(data_raw.get("filter_overlong_prompts", True)),
        truncation=str(data_raw.get("truncation", DataConfig.truncation)),
        shuffle=bool(data_raw.get("shuffle", True)),
        validation_shuffle=bool(data_raw.get("validation_shuffle", DataConfig.validation_shuffle)),
        seed=int(data_raw.get("seed", DataConfig.seed)),
        return_raw_chat=bool(data_raw.get("return_raw_chat", True)),
        enable_thinking=bool(data_raw.get("enable_thinking", False)),
        need_tools_kwargs=bool(data_raw.get("need_tools_kwargs", False)),
    )
    primary_teacher_raw = model_raw.get(
        "primary_teacher_path",
        model_raw.get("reasoning_teacher_path", model_raw.get("math_teacher_path")),
    )
    if primary_teacher_raw is None:
        raise ValueError(
            "Expected 'model.primary_teacher_path', 'model.reasoning_teacher_path', "
            "or 'model.math_teacher_path'."
        )
    secondary_teacher_raw = model_raw.get(
        "secondary_teacher_path",
        model_raw.get("code_teacher_path", primary_teacher_raw),
    )
    code_teacher_raw = model_raw.get(
        "code_teacher_path",
        secondary_teacher_raw if secondary_teacher_raw is not None else primary_teacher_raw,
    )
    if _same_model_path(secondary_teacher_raw, primary_teacher_raw):
        secondary_teacher_raw = (
            None if _same_model_path(code_teacher_raw, primary_teacher_raw) else code_teacher_raw
        )
    teacher_model_device = str(model_raw.get("teacher_model_device", "cpu")).lower()
    if teacher_model_device == "cuda":
        teacher_model_device = "gpu"
    if teacher_model_device not in {"cpu", "gpu"}:
        raise ValueError("Expected model.teacher_model_device to be one of: 'cpu', 'gpu', or 'cuda'.")
    attn_implementation = str(
        model_raw.get("attn_implementation", ModelConfig.attn_implementation)
    ).strip()
    if attn_implementation not in {"eager", "sdpa", "flash_attention_2"}:
        raise ValueError(
            "Expected model.attn_implementation to be one of: "
            "'flash_attention_2', 'sdpa', or 'eager'."
        )
    model = ModelConfig(
        student_path=str(model_raw["student_path"]),
        student_base_path=(
            None
            if model_raw.get("student_base_path", model_raw["student_path"]) is None
            else str(model_raw.get("student_base_path", model_raw["student_path"]))
        ),
        math_teacher_path=str(model_raw.get("math_teacher_path", primary_teacher_raw)),
        code_teacher_path=str(code_teacher_raw),
        reasoning_teacher_path=(
            None
            if model_raw.get("reasoning_teacher_path") is None
            else str(model_raw["reasoning_teacher_path"])
        ),
        primary_teacher_path=str(primary_teacher_raw),
        secondary_teacher_path=(None if secondary_teacher_raw is None else str(secondary_teacher_raw)),
        teacher_model_device=teacher_model_device,
        attn_implementation=attn_implementation,
        use_remove_padding=bool(
            model_raw.get("use_remove_padding", ModelConfig.use_remove_padding)
        ),
        reference_param_offload=bool(
            model_raw.get(
                "reference_param_offload",
                ModelConfig.reference_param_offload,
            )
        ),
    )

    config = MOPDConfig(
        data=data,
        model=model,
        actor=ActorConfig(**_expect_mapping(root.get("actor", {}), "actor")),
        rollout=RolloutConfig(**_expect_mapping(root.get("rollout", {}), "rollout")),
        rollout_correction=RolloutCorrectionConfig(
            **_expect_mapping(root.get("rollout_correction", {}), "rollout_correction")
        ),
        audit=AuditConfig(**_expect_mapping(root.get("audit", {}), "audit")),
        paper_eval=PaperEvalConfig(
            enabled=bool(paper_eval_raw.get("enabled", PaperEvalConfig.enabled)),
            script_path=_optional_string(paper_eval_raw.get("script_path"), "paper_eval.script_path"),
            model_path=_optional_string(paper_eval_raw.get("model_path"), "paper_eval.model_path"),
            output_dir=str(paper_eval_raw.get("output_dir", PaperEvalConfig.output_dir)),
            datasets=_string_list(
                paper_eval_raw.get("datasets", DEFAULT_PAPER_EVAL_DATASETS),
                "paper_eval.datasets",
            ),
            run_on_initial_validation=bool(
                paper_eval_raw.get("run_on_initial_validation", PaperEvalConfig.run_on_initial_validation)
            ),
            evaluate_current_checkpoint=bool(
                paper_eval_raw.get("evaluate_current_checkpoint", PaperEvalConfig.evaluate_current_checkpoint)
            ),
            fail_on_error=bool(paper_eval_raw.get("fail_on_error", PaperEvalConfig.fail_on_error)),
            timeout_seconds=int(paper_eval_raw.get("timeout_seconds", PaperEvalConfig.timeout_seconds)),
        ),
        trainer=TrainerConfig(**_expect_mapping(root.get("trainer", {}), "trainer")),
        ray_kwargs=RayKwargsConfig(
            ray_init=RayInitConfig(**_expect_mapping(root.get("ray_kwargs", {}).get("ray_init", {}), "ray_init"))
        ),
        runtime=RuntimeConfig(**_expect_mapping(root.get("runtime", {}), "runtime")),
        extra_overrides=_string_list(root.get("extra_overrides", []), "extra_overrides"),
    )
    validate_mopd_config(config)
    return config
