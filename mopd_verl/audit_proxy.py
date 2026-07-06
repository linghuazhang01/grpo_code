"""Batch metadata extraction helpers for MOPD audit logging."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


DOMAIN_LABEL_KEYS = ("opd_teacher", "domain", "source_domain", "ability")


def _non_tensor_list(value: Any, length: int, default: Any = None) -> list[Any]:
    if value is None:
        return [default for _ in range(length)]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value for _ in range(length)]


def _label_from_extra_info(extra_info: Any) -> Any:
    if isinstance(extra_info, str):
        try:
            extra_info = json.loads(extra_info)
        except json.JSONDecodeError:
            return None
    if not isinstance(extra_info, dict):
        return None
    for key in DOMAIN_LABEL_KEYS:
        value = extra_info.get(key)
        if value is not None:
            return value
    return None


def extract_teacher_domains(non_tensor: dict[str, Any], batch_size: int) -> list[str]:
    for key in DOMAIN_LABEL_KEYS:
        labels = _non_tensor_list(non_tensor.get(key), batch_size)
        if not all(label is None for label in labels):
            return [str(label if label is not None else "unknown") for label in labels]
    extra_infos = _non_tensor_list(non_tensor.get("extra_info"), batch_size)
    labels = [_label_from_extra_info(extra_info) for extra_info in extra_infos]
    if not all(label is None for label in labels):
        return [str(label if label is not None else "unknown") for label in labels]
    return ["unknown" for _ in range(batch_size)]


def extract_validation_datasets(non_tensor: dict[str, Any], batch_size: int) -> list[str]:
    explicit = _non_tensor_list(non_tensor.get("validation_dataset"), batch_size)
    data_sources = _non_tensor_list(non_tensor.get("data_source"), batch_size)
    abilities = _non_tensor_list(non_tensor.get("ability"), batch_size)
    labels: list[str] = []
    for idx in range(batch_size):
        if explicit[idx] is not None:
            labels.append(str(explicit[idx]))
            continue
        data_source = None if data_sources[idx] is None else str(data_sources[idx])
        ability = None if abilities[idx] is None else str(abilities[idx])
        if data_source:
            labels.append(data_source)
        elif ability in {"math", "code"}:
            labels.append(ability)
        else:
            labels.append("unknown")
    return labels


def extract_sample_ids(non_tensor: dict[str, Any], batch_size: int, step: int) -> list[str]:
    sample_ids = _non_tensor_list(non_tensor.get("sample_id"), batch_size)
    fallback_ids = _non_tensor_list(non_tensor.get("id"), batch_size)
    extra_infos = _non_tensor_list(non_tensor.get("extra_info"), batch_size)
    resolved = []
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
            resolved.append(f"step{step}:row{idx}")
    return resolved


def _mask_mean(matrix: Any, mask: Any) -> Any:
    denom = mask.sum(dim=-1).clamp(min=1)
    return (matrix * mask).sum(dim=-1) / denom


def response_mask_from_batch(tensor_batch: Any, reference: Any) -> Any:
    if "response_mask" in tensor_batch:
        return tensor_batch["response_mask"].detach().float()
    if "attention_mask" in tensor_batch:
        attention_mask = tensor_batch["attention_mask"].detach().float()
        if attention_mask.shape == reference.shape:
            return attention_mask
        if attention_mask.ndim == reference.ndim and attention_mask.shape[-1] >= reference.shape[-1]:
            return attention_mask[..., -reference.shape[-1] :]
    import torch

    return torch.ones_like(reference, dtype=torch.float32)
