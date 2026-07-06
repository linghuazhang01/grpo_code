"""Scalar validation and cost logging helpers for MOPD audit."""

from __future__ import annotations

from typing import Any

import numpy as np

from mopd_verl.audit_math import append_history, finite_float
from mopd_verl.tensorboard_tags import global_metric_category


def log_validation_metrics(logger: Any, val_metrics: dict[str, Any], step: int) -> dict[str, float]:
    if not logger.enabled or not logger.log_validation or not val_metrics:
        return {}

    rows = []
    scalar_metrics = {}
    for key, value in val_metrics.items():
        if logger._is_direct_audit_metric_key(str(key)):
            continue
        numeric = finite_float(value)
        if numeric is None:
            continue
        prev = logger._last_validation_metrics.get(key)
        gain = None if prev is None else numeric - prev
        logger._last_validation_metrics[key] = numeric
        safe_domain, safe_metric = logger._validation_tag_parts(key)
        if gain is not None:
            scalar_metrics[logger._domain_tag(safe_domain, "validation_gain", safe_metric)] = gain
            append_history(logger._validation_gain_history, key, gain, logger.tier2_window_size)
            gain_mean = _mean(logger._validation_gain_history.get(key, []))
            gain_variance = _var(logger._validation_gain_history.get(key, []))
            if gain_variance is not None:
                scalar_metrics[logger._domain_tag(safe_domain, "validation_gain_stats", "variance", safe_metric)] = (
                    gain_variance
                )
            if gain_mean is not None:
                scalar_metrics[logger._domain_tag(safe_domain, "validation_gain_stats", "mean", safe_metric)] = gain_mean
        rows.append(
            {
                "step": step,
                "metric_key": key,
                "metric_value": numeric,
                "previous_metric_value": prev,
                "gain": gain,
            }
        )
    logger._write_jsonl("validation_probe.jsonl", rows)
    if any(row["gain"] is not None for row in rows):
        logger._write_jsonl(
            "validation_gain_variance.jsonl",
            [
                {
                    "step": step,
                    "metric_key": row["metric_key"],
                    "gain_history": logger._validation_gain_history.get(row["metric_key"], []),
                    "gain_mean": _mean(logger._validation_gain_history.get(row["metric_key"], [])),
                    "gain_variance": _var(logger._validation_gain_history.get(row["metric_key"], [])),
                }
                for row in rows
                if row["gain"] is not None
            ],
        )
    return scalar_metrics


def log_training_cost(logger: Any, metrics: dict[str, Any], step: int, n_gpus: int = 1) -> dict[str, float]:
    if not logger.enabled:
        return {}

    step_seconds = finite_float(metrics.get("timing_s/step", metrics.get("perf/time_per_step")))
    total_tokens = finite_float(metrics.get("perf/total_num_tokens"))
    memory_peak = finite_float(metrics.get("perf/max_memory_allocated_gb", metrics.get("perf/max_memory_reserved_gb")))
    rows = [
        {
            "step": step,
            "gpu_seconds_step": None if step_seconds is None else step_seconds * max(1, n_gpus),
            "tokens_per_second": None
            if step_seconds is None or total_tokens is None
            else total_tokens / (step_seconds + 1e-8),
            "memory_peak_step": memory_peak,
            "step_seconds": step_seconds,
        }
    ]
    logger._write_jsonl("training_cost.jsonl", rows)
    scalar_metrics: dict[str, float] = {}
    for key, value in rows[0].items():
        numeric = finite_float(value)
        if numeric is not None:
            scalar_metrics[logger._global_tag(global_metric_category(key), key)] = numeric
    return scalar_metrics


def _var(values: list[float]) -> float | None:
    return float(np.var(values)) if values else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None
