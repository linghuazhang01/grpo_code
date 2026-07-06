"""TensorBoard scalar pruning for MOPD audit metrics."""

from __future__ import annotations

from typing import Any

from mopd_verl.audit_math import finite_float

UNPRUNED_MODES = {"", "none", "off", "false", "0", "full", "all"}
DIRECT_AUDIT_CATEGORIES = {
    "audit",
    "advantage",
    "cost",
    "full_grad",
    "full_grad_alignment",
    "full_grad_closure",
    "full_grad_contribution",
    "full_grad_conflict",
    "full_grad_cost",
    "full_grad_training_parity",
    "loss",
    "length",
    "optimization",
    "reward",
    "sample_grad",
    "sample_grad_closure",
    "sample_grad_contribution",
    "sample_grad_cos",
    "sample_grad_cost",
    "teacher",
    "entropy",
    "entropy_vocab_cosine",
    "token_conflict",
    "token_gap",
    "token_gap_vocab_cosine",
    "token_grad",
    "token_grad_closure",
    "token_grad_conflict",
    "token_grad_contribution",
    "token_grad_cost",
    "calibration",
    "coverage",
}

CORE_DOMAIN_DATA = {"domain_sample_count", "domain_token_count", "domain_token_frac"}
CORE_DOMAIN_LOSS = {
    "advantage_mean",
    "high_variance_sample_rate",
    "sample_opd_loss_mean",
    "sample_opd_loss_std",
    "sample_opd_loss_variance",
    "token_opd_loss_mean",
    "token_opd_loss_std",
    "token_opd_loss_variance",
}
CORE_DOMAIN_ADVANTAGE = {"positive_frac"}
CORE_DOMAIN_LENGTH = {"response_mean", "response_p95", "response_clip_ratio"}
CORE_SAMPLE_GRAD = {"norm_mean", "norm_p50", "norm_p95", "norm_max", "norm_cv", "sample_count"}
CORE_SAMPLE_GRAD_COS = {
    "all_parameters_disconnected_count",
    "attempted_count",
    "domain_cos_mean",
    "domain_cos_p05",
    "domain_cos_negative_frac",
    "sample_count",
    "unavailable_count",
    "valid_frac",
}
CORE_SAMPLE_GRAD_CONTRIBUTION = {
    "projection_share_mean",
    "projection_share_min",
    "projection_share_max",
    "projection_share_negative_frac",
    "projection_share_normalized_mean",
    "projection_share_normalized_min",
    "projection_share_normalized_max",
    "projection_share_normalized_negative_frac",
    "projection_share_normalized_sum",
    "projection_share_normalized_sum_error",
    "projection_share_sum",
    "projection_share_sum_across_replicas",
    "projection_share_sum_error",
    "projection_share_sum_raw",
    "projection_share_sum_raw_expected",
    "projection_share_sum_raw_error",
    "projection_share_scale_mean",
    "projection_share_replica_count",
    "projection_share_trusted",
    "top1_abs_share",
    "top1_abs_share_normalized",
}
CORE_SAMPLE_GRAD_CLOSURE = {
    "projection_share_normalized_sum",
    "projection_share_normalized_sum_error",
    "projection_share_sum",
    "projection_share_sum_error",
    "projection_share_sum_raw",
    "projection_share_sum_raw_expected",
    "projection_share_sum_raw_error",
    "valid_frac",
    "vector_available",
    "vector_candidate_norm",
    "vector_cosine",
    "vector_diff_norm",
    "vector_error_count",
    "vector_max_abs",
    "vector_norm_ratio",
    "vector_projection_share",
    "vector_reference_norm",
    "vector_rel_l2",
    "vector_slot_count",
}
CORE_SAMPLE_GRAD_COST = {
    "backward_recompute_count",
    "backward_sync_count",
    "restore_post_target_rel_l2_max",
    "restore_pre_target_rel_l2_max",
    "seconds_mean",
    "seconds_sum",
}
CORE_FULL_GRAD_CLOSURE = {
    "cosine",
    "max_abs",
    "norm_ratio",
    "projection_share",
    "rel_l2",
}
CORE_GLOBAL_LOSS = {
    "sample_opd_loss_mean",
    "sample_opd_loss_std",
    "sample_opd_loss_variance",
    "token_opd_loss_mean",
    "token_opd_loss_std",
    "token_opd_loss_variance",
}
CORE_GLOBAL_OPTIMIZATION = {"learning_rate"}
CORE_DOMAIN_TEACHER = {"teacher_confidence_mean", "teacher_student_gap_mean"}
CORE_TOKEN_GAP = {
    "gap_abs_mean",
    "gap_abs_p95",
    "gap_abs_sum",
    "gap_signed_mean",
    "gap_signed_p05",
    "gap_signed_p50",
    "gap_signed_p95",
}
CORE_TOKEN_GAP_VOCAB_COSINE = {
    "gap_abs_sum_cosine",
    "gap_signed_sum_cosine",
}
CORE_ENTROPY_VOCAB_COSINE = {
    "student_entropy_sum_cosine",
    "teacher_student_cross_entropy_sum_cosine",
}
CORE_DOMAIN_ENTROPY = {
    "cross_entropy_available",
    "entropy_distribution_available",
    "student_entropy_mean",
    "student_entropy_p50",
    "student_entropy_p95",
    "student_entropy_std",
    "sum_student_entropy",
    "sum_teacher_entropy",
    "sum_teacher_student_cross_entropy",
    "teacher_entropy_mean",
    "teacher_entropy_p50",
    "teacher_entropy_p95",
    "teacher_entropy_std",
    "teacher_student_cross_entropy_mean",
    "teacher_student_cross_entropy_p50",
    "teacher_student_cross_entropy_p95",
    "teacher_student_cross_entropy_std",
}
CORE_TOKEN_CONFLICT = {
    "combined_diff_mass",
    "combined_diff_mean",
    "combined_diff_mass_frac",
    "combined_diff_p95",
    "combined_diff_max",
    "opd_signal_abs_mean",
    "proxy_mass",
    "proxy_mean",
    "proxy_mass_frac",
    "student_teacher_diff_mass",
    "student_teacher_diff_mean",
    "student_teacher_diff_p95",
    "student_teacher_diff_max",
    "teacher_disagreement_mean",
    "teacher_teacher_diff_mass",
    "teacher_teacher_diff_mean",
    "teacher_teacher_diff_mass_frac",
    "teacher_teacher_diff_p95",
    "teacher_teacher_diff_max",
    "token_abs_opd_loss_mean",
    "top1_teacher_diff_share",
    "top10_teacher_diff_share",
    "top1_token_share",
    "top10_token_share",
    "unique_token_count",
}
CORE_TOKEN_GRAD = {
    "global_candidate_gap_mass",
    "global_candidate_gap_abs_mass",
    "global_candidate_loss_abs_mass",
    "global_candidate_sample_count",
    "global_candidate_token_count",
    "norm_mean",
    "norm_p95",
    "norm_max",
    "selected_sample_count",
    "selected_token_count",
}
CORE_TOKEN_GRAD_CONFLICT = {
    "conflict_to_other_max",
    "conflict_to_other_mean",
    "other_cos_mean",
    "other_cos_negative_frac",
    "other_cos_p05",
}
CORE_TOKEN_GRAD_CONTRIBUTION = {
    "negative_other_projection_share_sum",
    "other_projection_share_mean",
    "own_projection_share_mean",
    "own_projection_share_sum",
}
CORE_TOKEN_GRAD_CLOSURE = {
    "candidate_token_frac",
    "candidate_sample_frac",
    "selected_all_tokens",
    "selected_all_samples",
    "projection_share_error",
    "cosine_error",
    "norm_ratio",
    "norm_ratio_error",
}
CORE_TOKEN_GRAD_COST = {
    "autograd_seconds_sum",
    "available_token_count",
    "backward_fallback_count",
    "backward_fallback_seconds_sum",
    "global_candidate_gap_mass",
    "global_candidate_gap_abs_mass",
    "global_candidate_loss_abs_mass",
    "global_candidate_sample_count",
    "global_candidate_token_count",
    "max_memory_allocated_gb",
    "restore_original_max_abs_max",
    "restore_original_rel_l2_max",
    "restore_post_target_max_abs_max",
    "restore_post_target_rel_l2_max",
    "restore_pre_target_max_abs_max",
    "restore_pre_target_rel_l2_max",
    "seconds",
    "seconds_mean",
    "seconds_per_selected_token",
    "seconds_sum",
    "selected_sample_count",
    "selected_token_count",
    "unavailable_token_count",
    "valid_frac",
}
CORE_DOMAIN_REWARD = {"training_accuracy", "training_reward_mean"}
CORE_DOMAIN_COVERAGE = {"duplicate_rate"}
CORE_FULL_GRAD = {"grad_norm", "sample_count"}
CORE_FULL_GRAD_ALIGNMENT = {"full_grad_cosine_domain_total"}
CORE_FULL_GRAD_CONTRIBUTION = {"signed_projection_share"}
CORE_FULL_GRAD_TRAINING_PARITY = {
    "candidate_norm",
    "cosine",
    "diff_norm",
    "max_abs",
    "norm_ratio",
    "projection_share",
    "reference_norm",
    "rel_l2",
}
CORE_FULL_GRAD_TRAINING_PARITY_GROUPS = {
    "audit_total_vs_training_total",
    "sequence_total_vs_training_total",
}
CORE_CONFLICT = {
    "conflict_magnitude_i_k",
    "full_grad_cosine_train_i_k",
}
CORE_AUDIT = {
    "error",
    "full_gradient_autograd_unavailable",
    "full_gradient_domain_direct_recompute_available",
    "full_gradient_domain_direct_recompute_error",
    "full_gradient_domain_direct_recompute_used",
    "full_gradient_execution_timing_pre_update",
    "full_gradient_domain_sequential_available",
    "full_gradient_domain_sequential_unsupported",
    "full_gradient_replicated_all_reduce",
    "full_gradient_replica_count",
    "full_gradient_true_backward_fallback",
    "pre_update_audit_used",
    "sample_gradient_distributed_unsupported",
    "sample_gradient_cos_distributed_unsupported",
    "sample_gradient_norm_distributed_unsupported",
    "sample_gradient_distributed_world_size",
    "sample_gradient_zero_norm_count",
    "token_gradient_distributed_unsupported",
    "wall_time_step",
}
CORE_GLOBAL_DATA = {"domain_mix_entropy", "total_samples", "total_tokens"}
CORE_GLOBAL_COST = {"gpu_seconds_step", "memory_peak_step", "step_seconds", "tokens_per_second"}
CORE_ACTOR = {
    "actor/entropy",
    "actor/grad_norm",
    "actor/lr",
    "actor/pg_clipfrac",
    "actor/pg_loss",
    "actor/ppo_kl",
    "actor/tail_student_mass_on_teacher_ids",
    "actor/tail_teacher_mass",
    "actor/topk_distill_loss",
    "actor/topk_distill_weight",
    "actor/topk_student_mass_on_teacher_ids",
    "actor/topk_teacher_mass",
}
CORE_ROLLOUT_CORR = {
    "kl",
    "ppl_ratio",
    "rollout_is_catastrophic_token_fraction",
    "rollout_is_eff_sample_size",
    "rollout_is_veto_fraction",
}
CORE_LENGTH = {"clip_ratio", "max", "mean"}
CORE_TIMING_SECONDS = {"gen", "step", "testing", "update_actor"}
CORE_TIMING_PER_TOKEN = {"gen", "update_actor"}
CORE_PERF = {"max_memory_allocated_gb", "throughput", "time_per_step", "total_num_tokens"}
CORE_TRAINING = {"epoch", "global_step"}


def is_direct_audit_metric_key(key: str) -> bool:
    parts = _parts(key)
    return len(parts) >= 2 and parts[1] in DIRECT_AUDIT_CATEGORIES


def filter_tensorboard_metrics(metrics: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode.lower() in UNPRUNED_MODES:
        return metrics
    if not metrics:
        return {}

    filtered: dict[str, float] = {}
    for key, value in metrics.items():
        numeric = finite_float(value)
        if numeric is not None and keep_core_metric(str(key)):
            filtered[str(key)] = numeric
    return filtered


def keep_core_metric(key: str) -> bool:
    parts = _parts(key)
    if not parts:
        return False
    root = parts[0]
    category = parts[1] if len(parts) > 1 else ""
    metric = parts[-1]

    if key.startswith("val-core/") or key in CORE_ACTOR:
        return True
    if root == "critic":
        return len(parts) == 3 and parts[1] in {"advantages", "returns", "rewards", "score"} and metric == "mean"
    if root == "rollout_corr":
        return metric in CORE_ROLLOUT_CORR
    if root == "response_length_non_aborted":
        return metric == "mean"
    if root in {"prompt_length", "response_length"}:
        return metric in CORE_LENGTH
    if key == "response/aborted_ratio":
        return True
    if root == "timing_s":
        return metric in CORE_TIMING_SECONDS
    if root == "timing_per_token_ms":
        return metric in CORE_TIMING_PER_TOKEN
    if root == "perf":
        return metric in CORE_PERF
    if root == "training":
        return metric in CORE_TRAINING
    if root == "global":
        return _keep_global(category, metric, parts)
    return _keep_domain(category, metric, parts)


def _keep_global(category: str, metric: str, parts: list[str]) -> bool:
    if category == "audit":
        return metric in CORE_AUDIT
    if category == "cost":
        return metric in CORE_GLOBAL_COST
    if category == "full_grad_cost":
        return metric in {
            "backward_seconds",
            "domain_direct_recompute_seconds",
            "domain_summary_seconds",
            "finish_mini_batch_seconds",
            "max_memory_allocated_gb",
        }
    if category == "full_grad_closure":
        return metric in CORE_FULL_GRAD_CLOSURE
    if category == "token_grad_cost":
        return metric in CORE_TOKEN_GRAD_COST
    if category == "full_grad_alignment":
        return metric in CORE_FULL_GRAD_ALIGNMENT
    if category == "full_grad_contribution":
        return metric in CORE_FULL_GRAD_CONTRIBUTION
    if category == "full_grad_training_parity":
        return (
            len(parts) >= 4
            and parts[2] in CORE_FULL_GRAD_TRAINING_PARITY_GROUPS
            and metric in CORE_FULL_GRAD_TRAINING_PARITY
        )
    if category == "data":
        return metric in CORE_GLOBAL_DATA
    if category == "full_grad_conflict":
        return metric in CORE_CONFLICT
    if category == "loss":
        return metric in CORE_GLOBAL_LOSS
    if category == "token_gap_vocab_cosine":
        return metric in CORE_TOKEN_GAP_VOCAB_COSINE
    if category == "entropy_vocab_cosine":
        return metric in CORE_ENTROPY_VOCAB_COSINE
    if category == "optimization":
        return metric in CORE_GLOBAL_OPTIMIZATION
    if category == "validation":
        return False
    if category == "validation_gain":
        return not _contains_audit_category(parts[2:])
    if category == "validation_gain_stats":
        return metric in {"mean", "variance"} and not _contains_audit_category(parts[2:])
    return False


def _keep_domain(category: str, metric: str, parts: list[str]) -> bool:
    if category == "data":
        return metric in CORE_DOMAIN_DATA
    if category == "loss":
        return metric in CORE_DOMAIN_LOSS
    if category == "advantage":
        return metric in CORE_DOMAIN_ADVANTAGE
    if category == "length":
        return metric in CORE_DOMAIN_LENGTH
    if category == "sample_grad":
        return metric in CORE_SAMPLE_GRAD
    if category == "sample_grad_cos":
        return metric in CORE_SAMPLE_GRAD_COS
    if category == "sample_grad_contribution":
        return metric in CORE_SAMPLE_GRAD_CONTRIBUTION
    if category == "sample_grad_closure":
        return metric in CORE_SAMPLE_GRAD_CLOSURE
    if category == "sample_grad_cost":
        return metric in CORE_SAMPLE_GRAD_COST
    if category == "full_grad":
        return metric in CORE_FULL_GRAD
    if category == "teacher":
        return metric in CORE_DOMAIN_TEACHER
    if category == "token_gap":
        return metric in CORE_TOKEN_GAP
    if category == "entropy":
        return metric in CORE_DOMAIN_ENTROPY
    if category == "token_conflict":
        return metric in CORE_TOKEN_CONFLICT
    if category == "token_grad":
        return metric in CORE_TOKEN_GRAD or metric.endswith(
            (
                "_cos_to_domain",
                "_gap_mass",
                "_gap_mass_frac",
                "_gap_abs_mass",
                "_gap_abs_mass_frac",
                "_loss_abs_mass",
                "_loss_abs_mass_frac",
                "_non_none_grad_count",
                "_none_grad_count",
                "_param_count",
                "_score_mass",
                "_score_mass_frac",
                "_selected_sample_count",
                "_selected_token_count",
            )
        )
    if category == "token_grad_conflict":
        return metric in CORE_TOKEN_GRAD_CONFLICT
    if category == "token_grad_contribution":
        return metric in CORE_TOKEN_GRAD_CONTRIBUTION or metric.endswith("_projection_share")
    if category == "token_grad_closure":
        return any(metric.endswith(f"_{name}") for name in CORE_TOKEN_GRAD_CLOSURE)
    if category == "token_grad_cost":
        return metric in CORE_TOKEN_GRAD_COST
    if category == "reward":
        return metric in CORE_DOMAIN_REWARD
    if category == "calibration":
        return metric == "calibration_error"
    if category == "coverage":
        return metric in CORE_DOMAIN_COVERAGE
    if category == "validation":
        return False
    if category == "validation_gain":
        return not _contains_audit_category(parts[2:])
    if category == "validation_gain_stats":
        return metric in {"mean", "variance"} and not _contains_audit_category(parts[2:])
    return False


def _contains_audit_category(parts: list[str]) -> bool:
    return any(part in DIRECT_AUDIT_CATEGORIES for part in parts)


def _parts(key: str) -> list[str]:
    return [part for part in key.replace("\\", "/").split("/") if part]
