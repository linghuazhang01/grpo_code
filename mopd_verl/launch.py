"""Build and run verl GRPO training commands."""

from __future__ import annotations

import argparse
import datetime
import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from mopd_verl.settings import MOPDConfig, load_config


def _bool(value: bool) -> str:
    return "True" if value else "False"


def _hydra_list(values: Sequence[str]) -> str:
    quoted = ", ".join(f"'{value}'" for value in values)
    return f"[{quoted}]"


def _hydra_float_dict(values: Mapping[str, float]) -> str:
    items = ", ".join(f"{key}: {float(value):g}" for key, value in values.items())
    return "{" + items + "}"


def _hydra_list_dict(values: Mapping[str, Sequence[str]]) -> str:
    items = ", ".join(f"{key}: {_hydra_list(file_paths)}" for key, file_paths in values.items())
    return "{" + items + "}"


def _hydra_scalar(value: object) -> str:
    if value is None:
        return "null"
    return str(value)


def _rollout_multiturn_overrides(config: MOPDConfig) -> list[str]:
    rollout = config.rollout
    if not rollout.multi_turn_enable and rollout.multi_turn_tool_config_path is None:
        return []

    overrides = [
        f"actor_rollout_ref.rollout.multi_turn.enable={str(rollout.multi_turn_enable).lower()}",
        f"actor_rollout_ref.rollout.multi_turn.max_parallel_calls={rollout.multi_turn_max_parallel_calls}",
        f"actor_rollout_ref.rollout.multi_turn.max_tool_response_length={rollout.multi_turn_max_tool_response_length}",
        "actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side="
        f"{rollout.multi_turn_tool_response_truncate_side}",
        f"actor_rollout_ref.rollout.multi_turn.format={rollout.multi_turn_format}",
        "actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode="
        f"{rollout.multi_turn_tokenization_sanity_check_mode}",
    ]
    if rollout.multi_turn_tool_config_path is not None:
        overrides.append(f"actor_rollout_ref.rollout.multi_turn.tool_config_path={rollout.multi_turn_tool_config_path}")
    if rollout.multi_turn_max_assistant_turns is not None:
        overrides.append(
            f"actor_rollout_ref.rollout.multi_turn.max_assistant_turns={rollout.multi_turn_max_assistant_turns}"
        )
    if rollout.multi_turn_max_user_turns is not None:
        overrides.append(f"actor_rollout_ref.rollout.multi_turn.max_user_turns={rollout.multi_turn_max_user_turns}")
    return overrides


def _audit_overrides(config: MOPDConfig) -> list[str]:
    audit = config.audit
    if not audit.enabled:
        return []

    return [
        f"+mopd_audit.enabled={str(audit.enabled).lower()}",
        f"+mopd_audit.output_dir={audit.output_dir}",
        f"+mopd_audit.domains={_hydra_list(audit.domains)}",
        f"+mopd_audit.tensorboard_prefix={audit.tensorboard_prefix}",
        f"+mopd_audit.tensorboard_layout={audit.tensorboard_layout}",
        f"+mopd_audit.tensorboard_prune_mode={audit.tensorboard_prune_mode}",
        f"+mopd_audit.loss_variance_signal={audit.loss_variance_signal}",
        f"+mopd_audit.max_samples_per_domain={_hydra_scalar(audit.max_samples_per_domain)}",
        f"+mopd_audit.high_variance_cv_threshold={audit.high_variance_cv_threshold}",
        f"+mopd_audit.log_sample_level={str(audit.log_sample_level).lower()}",
        f"+mopd_audit.log_sample_level_freq_steps={audit.log_sample_level_freq_steps}",
        f"+mopd_audit.log_validation_metrics={str(audit.log_validation_metrics).lower()}",
        f"+mopd_audit.log_validation_metrics_freq_steps={audit.log_validation_metrics_freq_steps}",
        f"+mopd_audit.tier2_window_size={audit.tier2_window_size}",
        f"+mopd_audit.calibration_bins={audit.calibration_bins}",
        f"+mopd_audit.full_gradient_enabled={str(audit.full_gradient_enabled).lower()}",
        f"+mopd_audit.full_gradient_freq_steps={audit.full_gradient_freq_steps}",
        f"+mopd_audit.full_grad_training_parity_freq_steps={audit.full_grad_training_parity_freq_steps}",
        "+mopd_audit.full_gradient_train_max_samples_per_domain="
        f"{_hydra_scalar(audit.full_gradient_train_max_samples_per_domain)}",
        f"+mopd_audit.full_gradient_micro_batch_size_per_gpu={audit.full_gradient_micro_batch_size_per_gpu}",
        f"+mopd_audit.full_gradient_storage_dtype={audit.full_gradient_storage_dtype}",
        "+mopd_audit.full_gradient_direct_recompute_enabled="
        f"{str(audit.full_gradient_direct_recompute_enabled).lower()}",
        "+mopd_audit.sequence_masked_target_enabled="
        f"{str(audit.sequence_masked_target_enabled).lower()}",
        "+mopd_audit.sequence_masked_target_use_as_primary="
        f"{str(audit.sequence_masked_target_use_as_primary).lower()}",
        f"+mopd_audit.sample_gradient_enabled={str(audit.sample_gradient_enabled).lower()}",
        f"+mopd_audit.sample_gradient_freq_steps={audit.sample_gradient_freq_steps}",
        f"+mopd_audit.sample_gradient_norm_enabled={str(audit.sample_gradient_norm_enabled).lower()}",
        f"+mopd_audit.sample_gradient_cos_enabled={str(audit.sample_gradient_cos_enabled).lower()}",
        f"+mopd_audit.sample_gradient_cos_freq_steps={audit.sample_gradient_cos_freq_steps}",
        "+mopd_audit.sample_gradient_backward_recompute_enabled="
        f"{str(audit.sample_gradient_backward_recompute_enabled).lower()}",
        "+mopd_audit.sample_gradient_backward_sync_enabled="
        f"{str(audit.sample_gradient_backward_sync_enabled).lower()}",
        f"+mopd_audit.sample_gradient_log_sample_level={str(audit.sample_gradient_log_sample_level).lower()}",
        "+mopd_audit.sample_gradient_log_sample_level_freq_steps="
        f"{audit.sample_gradient_log_sample_level_freq_steps}",
        f"+mopd_audit.full_gradient_offload_domain_gradients={str(audit.full_gradient_offload_domain_gradients).lower()}",
        f"+mopd_audit.token_gap_enabled={str(audit.token_gap_enabled).lower()}",
        f"+mopd_audit.token_gap_freq_steps={audit.token_gap_freq_steps}",
        f"+mopd_audit.token_gap_vocab_vector_enabled={str(audit.token_gap_vocab_vector_enabled).lower()}",
        f"+mopd_audit.token_gap_vocab_vector_freq_steps={audit.token_gap_vocab_vector_freq_steps}",
        f"+mopd_audit.token_gap_vocab_size={_hydra_scalar(audit.token_gap_vocab_size)}",
        f"+mopd_audit.entropy_enabled={str(audit.entropy_enabled).lower()}",
        f"+mopd_audit.entropy_freq_steps={audit.entropy_freq_steps}",
        f"+mopd_audit.entropy_vocab_vector_enabled={str(audit.entropy_vocab_vector_enabled).lower()}",
        f"+mopd_audit.entropy_vocab_vector_freq_steps={audit.entropy_vocab_vector_freq_steps}",
        f"+mopd_audit.token_conflict_enabled={str(audit.token_conflict_enabled).lower()}",
        f"+mopd_audit.token_conflict_freq_steps={audit.token_conflict_freq_steps}",
        f"+mopd_audit.token_conflict_top_k={_hydra_scalar(audit.token_conflict_top_k)}",
        f"+mopd_audit.token_gradient_enabled={str(audit.token_gradient_enabled).lower()}",
        f"+mopd_audit.token_gradient_freq_steps={audit.token_gradient_freq_steps}",
        f"+mopd_audit.token_gradient_gap_selection_enabled="
        f"{str(audit.token_gradient_gap_selection_enabled).lower()}",
        f"+mopd_audit.token_gradient_gap_abs_selection_enabled="
        f"{str(audit.token_gradient_gap_abs_selection_enabled).lower()}",
        f"+mopd_audit.token_gradient_loss_abs_selection_enabled="
        f"{str(audit.token_gradient_loss_abs_selection_enabled).lower()}",
        f"+mopd_audit.token_gradient_top_k={audit.token_gradient_top_k}",
        f"+mopd_audit.token_gradient_top_p={audit.token_gradient_top_p}",
        f"+mopd_audit.token_gradient_strict_grad_restore="
        f"{str(audit.token_gradient_strict_grad_restore).lower()}",
        "+mopd_audit.token_gradient_backward_recompute_enabled="
        f"{str(audit.token_gradient_backward_recompute_enabled).lower()}",
        "+mopd_audit.token_gradient_backward_sync_enabled="
        f"{str(audit.token_gradient_backward_sync_enabled).lower()}",
    ]


def _paper_eval_overrides(config: MOPDConfig) -> list[str]:
    paper_eval = config.paper_eval
    if not paper_eval.enabled:
        return []

    return [
        f"+paper_eval.enabled={str(paper_eval.enabled).lower()}",
        f"+paper_eval.script_path={_hydra_scalar(paper_eval.script_path)}",
        f"+paper_eval.model_path={_hydra_scalar(paper_eval.model_path)}",
        f"+paper_eval.output_dir={paper_eval.output_dir}",
        f"+paper_eval.datasets={_hydra_list(paper_eval.datasets)}",
        f"+paper_eval.run_on_initial_validation={str(paper_eval.run_on_initial_validation).lower()}",
        f"+paper_eval.evaluate_current_checkpoint={str(paper_eval.evaluate_current_checkpoint).lower()}",
        f"+paper_eval.fail_on_error={str(paper_eval.fail_on_error).lower()}",
        f"+paper_eval.timeout_seconds={paper_eval.timeout_seconds}",
    ]


def build_overrides(config: MOPDConfig) -> list[str]:
    data = config.data
    model = config.model
    actor = config.actor
    rollout = config.rollout
    rollout_correction = config.rollout_correction
    trainer = config.trainer
    ray_init = config.ray_kwargs.ray_init

    ray_overrides = []
    if ray_init.include_dashboard is not None:
        ray_overrides.append(f"+ray_kwargs.ray_init.include_dashboard={_bool(ray_init.include_dashboard)}")
    if ray_init.num_cpus is not None:
        ray_overrides.append(f"ray_kwargs.ray_init.num_cpus={ray_init.num_cpus}")

    vllm_engine_overrides = []
    if rollout.num_gpu_blocks_override is not None:
        vllm_engine_overrides.append(
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.num_gpu_blocks_override="
            f"{rollout.num_gpu_blocks_override}"
        )
    if rollout.max_model_len is not None:
        vllm_engine_overrides.append(f"actor_rollout_ref.rollout.max_model_len={rollout.max_model_len}")

    domain_sampling_overrides = []
    if data.domain_train_files:
        domain_sampling_overrides.append(f"+data.domain_train_files={_hydra_list_dict(data.domain_train_files)}")
        domain_sampling_overrides.append(
            f"+data.domain_sampling_replacement={str(data.domain_sampling_replacement).lower()}"
        )
    if data.domain_sampling_weights:
        domain_sampling_overrides.append(
            f"+data.domain_sampling_weights={_hydra_float_dict(data.domain_sampling_weights)}"
        )

    model_overrides = [f"actor_rollout_ref.model.path={model.student_path}"]
    if model.student_base_path is not None:
        model_overrides.append(f"+actor_rollout_ref.model.base_model_path={model.student_base_path}")
    model_overrides.append(f"+actor_rollout_ref.ref.model.path={model.primary_teacher_path}")
    model_overrides.append(f"+actor_rollout_ref.ref.model.teacher_model_device={model.teacher_model_device}")
    if model.secondary_teacher_path is not None:
        model_overrides.append(f"+actor_rollout_ref.ref.model.base_model_path={model.secondary_teacher_path}")

    overrides = [
        "algorithm.adv_estimator=grpo",
        f"algorithm.rollout_correction.rollout_is={rollout_correction.rollout_is}",
        f"algorithm.rollout_correction.rollout_is_threshold={rollout_correction.rollout_is_threshold}",
        f"algorithm.rollout_correction.rollout_rs={_hydra_scalar(rollout_correction.rollout_rs)}",
        f"algorithm.rollout_correction.bypass_mode={str(rollout_correction.bypass_mode).lower()}",
        f"actor_rollout_ref.rollout.calculate_log_probs={_bool(rollout.calculate_log_probs)}",
        f"data.train_files={_hydra_list(data.train_files)}",
        f"data.val_files={_hydra_list(data.val_files)}",
        f"data.train_batch_size={data.train_batch_size}",
        f"data.val_batch_size={_hydra_scalar(data.val_batch_size)}",
        f"data.max_prompt_length={data.max_prompt_length}",
        f"data.max_response_length={data.max_response_length}",
        f"data.filter_overlong_prompts={_bool(data.filter_overlong_prompts)}",
        f"data.truncation={data.truncation}",
        f"data.shuffle={_bool(data.shuffle)}",
        f"data.validation_shuffle={_bool(data.validation_shuffle)}",
        f"data.seed={data.seed}",
        f"data.return_raw_chat={_bool(data.return_raw_chat)}",
        f"+data.apply_chat_template_kwargs.enable_thinking={_bool(data.enable_thinking)}",
        f"+data.need_tools_kwargs={_bool(data.need_tools_kwargs)}",
        *model_overrides,
        f"actor_rollout_ref.actor.optim.lr={actor.learning_rate}",
        f"actor_rollout_ref.actor.optim.lr_warmup_steps_ratio={actor.lr_warmup_steps_ratio}",
        "actor_rollout_ref.model.use_remove_padding="
        f"{_bool(model.use_remove_padding)}",
        f"actor_rollout_ref.actor.policy_loss.only_reverse_kl_advantages={_bool(actor.only_reverse_kl_advantages)}",
        f"actor_rollout_ref.actor.policy_loss.lambda_vals={actor.lambda_vals}",
        f"actor_rollout_ref.actor.policy_loss.multi_teacher_distill={str(actor.multi_teacher_distill).lower()}",
        f"++actor_rollout_ref.actor.policy_loss.distill_loss_builder={actor.distill_loss_builder}",
        f"++actor_rollout_ref.actor.policy_loss.distill_mode={actor.distill_mode}",
        f"++actor_rollout_ref.actor.policy_loss.topk_distill_enabled={str(actor.topk_distill_enabled).lower()}",
        f"++actor_rollout_ref.actor.policy_loss.topk_distill_kl_direction={actor.topk_distill_kl_direction}",
        f"++actor_rollout_ref.actor.policy_loss.topk_distill_k={actor.topk_distill_k}",
        f"++actor_rollout_ref.actor.policy_loss.topk_distill_support_source={actor.topk_distill_support_source}",
        f"++actor_rollout_ref.actor.policy_loss.topk_distill_tail_bucket={str(actor.topk_distill_tail_bucket).lower()}",
        f"++actor_rollout_ref.actor.policy_loss.topk_distill_temperature={actor.topk_distill_temperature}",
        f"++actor_rollout_ref.actor.policy_loss.topk_distill_loss_weight={actor.topk_distill_loss_weight}",
        f"++actor_rollout_ref.actor.policy_loss.topk_distill_logprob_chunk_size={actor.topk_distill_logprob_chunk_size}",
        f"++actor_rollout_ref.actor.policy_loss.topk_distill_logprob_mode={actor.topk_distill_logprob_mode}",
        "++actor_rollout_ref.actor.policy_loss.teacher_prefix_enabled="
        f"{str(actor.teacher_prefix_enabled or rollout.teacher_prefix_sampling_enabled).lower()}",
        f"++actor_rollout_ref.actor.policy_loss.teacher_prefix_loss_region={actor.teacher_prefix_loss_region}",
        f"++actor_rollout_ref.actor.policy_loss.teacher_prefix_forward_kl_weight={actor.teacher_prefix_forward_kl_weight}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={actor.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={actor.ppo_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.actor.use_dynamic_bsz={_bool(actor.use_dynamic_bsz)}",
        f"actor_rollout_ref.actor.use_kl_loss={_bool(actor.use_kl_loss)}",
        f"actor_rollout_ref.actor.kl_loss_coef={actor.kl_loss_coef}",
        f"actor_rollout_ref.actor.kl_loss_type={actor.kl_loss_type}",
        f"actor_rollout_ref.actor.entropy_coeff={actor.entropy_coeff}",
        f"actor_rollout_ref.actor.ppo_max_token_len_per_gpu={actor.ppo_max_token_len_per_gpu}",
        f"actor_rollout_ref.model.enable_gradient_checkpointing={_bool(actor.gradient_checkpointing)}",
        "+actor_rollout_ref.model.override_config.attn_implementation="
        f"{model.attn_implementation}",
        f"actor_rollout_ref.actor.fsdp_config.param_offload={_bool(actor.param_offload)}",
        f"actor_rollout_ref.actor.fsdp_config.optimizer_offload={_bool(actor.optimizer_offload)}",
        f"actor_rollout_ref.actor.fsdp_config.model_dtype={actor.model_dtype}",
        *(
            [f"actor_rollout_ref.actor.fsdp_config.fsdp_size={actor.fsdp_size}"]
            if actor.fsdp_size is not None
            else []
        ),
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={rollout.log_prob_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={rollout.tensor_model_parallel_size}",
        f"actor_rollout_ref.rollout.name={rollout.name}",
        f"actor_rollout_ref.rollout.mode={rollout.mode}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={rollout.gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.enforce_eager={_bool(rollout.enforce_eager)}",
        f"actor_rollout_ref.rollout.enable_chunked_prefill={_bool(rollout.enable_chunked_prefill)}",
        f"actor_rollout_ref.rollout.n={rollout.n}",
        f"actor_rollout_ref.rollout.max_num_batched_tokens={rollout.max_num_batched_tokens}",
        f"actor_rollout_ref.rollout.max_num_seqs={rollout.max_num_seqs}",
        f"actor_rollout_ref.rollout.temperature={rollout.temperature}",
        f"actor_rollout_ref.rollout.top_p={rollout.top_p}",
        f"++actor_rollout_ref.rollout.teacher_prefix_sampling_enabled={_bool(rollout.teacher_prefix_sampling_enabled)}",
        f"++actor_rollout_ref.rollout.teacher_prefix_length={rollout.teacher_prefix_length}",
        f"++actor_rollout_ref.rollout.teacher_prefix_dataset_key={rollout.teacher_prefix_dataset_key}",
        f"actor_rollout_ref.rollout.val_kwargs.do_sample={_bool(rollout.val_do_sample)}",
        f"actor_rollout_ref.rollout.val_kwargs.temperature={rollout.val_temperature}",
        f"actor_rollout_ref.rollout.val_kwargs.top_p={rollout.val_top_p}",
        f"actor_rollout_ref.rollout.val_kwargs.n={rollout.val_n}",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.fsdp_config.param_offload="
        f"{_bool(model.reference_param_offload)}",
        "actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16",
        "algorithm.use_kl_in_reward=False",
        "reward_model.reward_manager=naive",
        f"trainer.critic_warmup={trainer.critic_warmup}",
        f"trainer.val_before_train={_bool(trainer.val_before_train)}",
        f"trainer.logger={trainer.logger}",
        f"trainer.log_val_generations={trainer.log_val_generations}",
        f"trainer.project_name={trainer.project_name}",
        f"trainer.experiment_name={trainer.experiment_name}",
        f"trainer.n_gpus_per_node={trainer.n_gpus_per_node}",
        f"trainer.nnodes={trainer.nnodes}",
        f"trainer.save_freq={trainer.save_freq}",
        f"trainer.default_local_dir={trainer.default_local_dir}",
        f"trainer.test_freq={trainer.test_freq}",
        f"trainer.total_epochs={trainer.total_epochs}",
        f"trainer.total_training_steps={_hydra_scalar(trainer.total_training_steps)}",
        f"trainer.max_actor_ckpt_to_keep={_hydra_scalar(trainer.max_actor_ckpt_to_keep)}",
        f"trainer.max_critic_ckpt_to_keep={_hydra_scalar(trainer.max_critic_ckpt_to_keep)}",
        f"trainer.resume_mode={trainer.resume_mode}",
    ]
    overrides.extend(ray_overrides)
    overrides.extend(vllm_engine_overrides)
    overrides.extend(domain_sampling_overrides)
    overrides.extend(_rollout_multiturn_overrides(config))
    overrides.extend(_audit_overrides(config))
    overrides.extend(_paper_eval_overrides(config))
    overrides.extend(config.extra_overrides)
    return overrides


def build_command(config: MOPDConfig, extra_args: Sequence[str] | None = None) -> list[str]:
    command = [config.runtime.python_bin, "-m", config.runtime.verl_module]
    command.extend(build_overrides(config))
    command.extend(extra_args or [])
    return command


def format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _valid_env_key(key: str) -> bool:
    if not key or key[0].isdigit():
        return False
    return all(char == "_" or char.isalnum() for char in key)


def _read_env_file(path: str | None) -> dict[str, str]:
    if not path:
        return {}

    env_path = Path(path).expanduser()
    if not env_path.is_absolute():
        env_path = Path(__file__).resolve().parents[1] / env_path
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"Invalid env file line {line_number} in {env_path}: expected KEY=value.")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not _valid_env_key(key):
            raise ValueError(f"Invalid env key {key!r} on line {line_number} in {env_path}.")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def run_command(command: Sequence[str], config: MOPDConfig) -> int:
    env = os.environ.copy()
    for key, value in _read_env_file(config.runtime.env_file).items():
        env.setdefault(key, value)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONINTMAXSTRDIGITS", "0")
    if config.runtime.wandb_entity is not None:
        env.setdefault("WANDB_ENTITY", config.runtime.wandb_entity)
    env.setdefault("WANDB_MODE", config.runtime.wandb_mode)
    env.setdefault("USED_MODEL", config.runtime.used_model)
    return subprocess.call(list(command), env=env)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "grpo" / "configs" / "m2rl_if_science_mix.yaml"),
        help="Path to a GRPO YAML config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the verl command without executing it.")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra Hydra overrides after '--'.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    extra_args = list(args.extra)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    # Append timestamp to experiment name so repeated runs produce distinct
    # tensorboard log dirs and don't collide.
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    extra_args = [f"trainer.experiment_name={config.trainer.experiment_name}_{timestamp}", *extra_args]

    command = build_command(config, extra_args)
    if args.dry_run:
        sys.stdout.write(format_command(command) + "\n")
        return 0
    return run_command(command, config)


if __name__ == "__main__":
    raise SystemExit(main())
