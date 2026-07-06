"""Numeric helpers for the MOPD audit logger."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def finite_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def ece(confidences: list[float], correctness: list[float], bins: int) -> float | None:
    if not confidences or not correctness:
        return None
    conf = np.clip(np.asarray(confidences, dtype=np.float64), 0.0, 1.0)
    corr = np.clip(np.asarray(correctness, dtype=np.float64), 0.0, 1.0)
    total = float(len(conf))
    error = 0.0
    for idx in range(max(1, bins)):
        lower = idx / max(1, bins)
        upper = (idx + 1) / max(1, bins)
        mask = (conf >= lower) & (conf <= upper if idx == bins - 1 else conf < upper)
        if np.any(mask):
            error += float(mask.sum()) / total * abs(float(corr[mask].mean()) - float(conf[mask].mean()))
    return float(error)


def append_history(history: dict[str, list[float]], key: str, value: float | None, max_len: int) -> None:
    if value is None:
        return
    values = history.setdefault(key, [])
    values.append(float(value))
    if len(values) > max_len:
        del values[: len(values) - max_len]
