"""TensorBoard tag naming helpers for MOPD audit metrics."""

from __future__ import annotations

from typing import Any


def safe_name(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _is_domain_data_metric(key: str) -> bool:
    return key in {
        "domain_sample_count",
        "domain_token_count",
        "domain_token_frac",
    }


def _is_domain_loss_metric(key: str) -> bool:
    return key in {
        "advantage_mean",
        "sample_opd_loss_mean",
        "sample_opd_loss_std",
        "sample_opd_loss_variance",
        "token_opd_loss_mean",
        "token_opd_loss_std",
        "token_opd_loss_variance",
        "high_variance_sample_rate",
    }


def _is_domain_advantage_metric(key: str) -> bool:
    return key in {"positive_frac"}


def _is_domain_length_metric(key: str) -> bool:
    return key in {"response_mean", "response_p95", "response_clip_ratio"}


def _is_domain_sample_grad_metric(key: str) -> bool:
    return key.startswith("norm_") or key in {"sample_count"}


def _is_domain_sample_grad_cos_metric(key: str) -> bool:
    return key.startswith("domain_cos_") or key in {
        "all_parameters_disconnected_count",
        "attempted_count",
        "sample_count",
        "unavailable_count",
        "valid_frac",
    }


def _is_domain_sample_grad_contribution_metric(key: str) -> bool:
    return key.startswith("projection_share_") or key == "top1_abs_share"


def _is_domain_teacher_metric(key: str) -> bool:
    return key in {"teacher_confidence_mean", "teacher_student_gap_mean"}


def _is_domain_token_conflict_metric(key: str) -> bool:
    return key in {
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


def _is_domain_token_gap_metric(key: str) -> bool:
    return key.startswith("gap_signed_") or key.startswith("gap_abs_")


def _is_domain_entropy_metric(key: str) -> bool:
    return (
        key.startswith("student_entropy_")
        or key.startswith("teacher_entropy_")
        or key.startswith("teacher_student_cross_entropy_")
        or key in {
            "cross_entropy_available",
            "entropy_distribution_available",
            "sum_student_entropy",
            "sum_teacher_entropy",
            "sum_teacher_student_cross_entropy",
        }
    )


def _is_domain_reward_metric(key: str) -> bool:
    return key in {"training_accuracy", "training_reward_mean"}


def domain_metric_category(key: str) -> str:
    if _is_domain_data_metric(key):
        return "data"
    if _is_domain_loss_metric(key):
        return "loss"
    if _is_domain_advantage_metric(key):
        return "advantage"
    if _is_domain_length_metric(key):
        return "length"
    if _is_domain_sample_grad_metric(key):
        return "sample_grad"
    if _is_domain_sample_grad_cos_metric(key):
        return "sample_grad_cos"
    if _is_domain_sample_grad_contribution_metric(key):
        return "sample_grad_contribution"
    if _is_domain_reward_metric(key):
        return "reward"
    if _is_domain_teacher_metric(key):
        return "teacher"
    if _is_domain_token_gap_metric(key):
        return "token_gap"
    if _is_domain_entropy_metric(key):
        return "entropy"
    if _is_domain_token_conflict_metric(key):
        return "token_conflict"
    if key.startswith("calibration"):
        return "calibration"
    if key == "duplicate_rate":
        return "coverage"
    return "misc"


def global_metric_category(key: str) -> str:
    if key in {"gpu_seconds_step", "tokens_per_second", "memory_peak_step", "step_seconds"}:
        return "cost"
    if key in {"total_tokens", "total_samples", "domain_mix_entropy"}:
        return "data"
    if key.startswith("audit_"):
        return "audit"
    return "misc"
