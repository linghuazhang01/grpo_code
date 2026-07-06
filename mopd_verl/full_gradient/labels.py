"""Label and sample metadata helpers for full-gradient audit."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from verl import DataProto


_TEACHER_LABEL_KEY = "opd_teacher"
_DOMAIN_LABEL_KEYS = ("domain", "source_domain", "ability", "data_source")


def _non_tensor_list(value: Any, length: int, default: Any = None) -> list[Any]:
    if length <= 0:
        return []
    if value is None:
        return [default for _ in range(length)]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            items = [value.item()]
        else:
            items = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]

    if not items:
        return [default for _ in range(length)]
    if len(items) == 1 and length > 1:
        return [items[0] for _ in range(length)]
    if len(items) < length:
        return items + [default for _ in range(length - len(items))]
    if len(items) > length:
        return items[:length]
    return items


def _label_from_extra_info(extra_info: Any) -> Any:
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            return None
    if not isinstance(extra_info, dict):
        return None
    for key in _DOMAIN_LABEL_KEYS:
        value = extra_info.get(key)
        if value is not None:
            return value
    return None


def _labels_from_mapping(mapping: dict[str, Any], batch_size: int) -> list[str]:
    for key in _DOMAIN_LABEL_KEYS:
        labels = _non_tensor_list(mapping.get(key), batch_size)
        if not all(label is None for label in labels):
            return [str(label if label is not None else "unknown") for label in labels]
    extra_infos = _non_tensor_list(mapping.get("extra_info"), batch_size)
    labels = [_label_from_extra_info(extra_info) for extra_info in extra_infos]
    if not all(label is None for label in labels):
        return [str(label if label is not None else "unknown") for label in labels]
    return ["unknown" for _ in range(batch_size)]


def _teacher_labels(data: DataProto) -> list[str]:
    """Return audit domain labels, not actor teacher labels."""
    return _labels_from_mapping(data.non_tensor_batch, len(data))


def _sample_ids(data: DataProto, step: int, fallback_prefix: str | None = None) -> list[str]:
    batch_size = len(data)
    sample_ids = _non_tensor_list(data.non_tensor_batch.get("sample_id"), batch_size)
    fallback_ids = _non_tensor_list(data.non_tensor_batch.get("id"), batch_size)
    extra_infos = _non_tensor_list(data.non_tensor_batch.get("extra_info"), batch_size)
    resolved: list[str] = []
    for idx, sample_id in enumerate(sample_ids):
        if sample_id is not None:
            resolved.append(str(sample_id))
        elif fallback_ids[idx] is not None:
            resolved.append(str(fallback_ids[idx]))
        elif isinstance(extra_infos[idx], dict) and extra_infos[idx].get("sample_id") is not None:
            resolved.append(str(extra_infos[idx]["sample_id"]))
        elif isinstance(extra_infos[idx], dict) and extra_infos[idx].get("id") is not None:
            resolved.append(str(extra_infos[idx]["id"]))
        else:
            prefix = fallback_prefix or f"step{step}"
            resolved.append(f"{prefix}:row{idx}")
    return resolved


def _response_token_count(data: DataProto) -> float:
    if data.batch is None or len(data) == 0:
        return 0.0
    if "response_mask" in data.batch:
        return float(data.batch["response_mask"].sum().item())
    if "responses" in data.batch:
        return float(data.batch["responses"].numel())
    return float(len(data))
