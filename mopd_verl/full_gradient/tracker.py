"""Full-parameter gradient audit helpers for patched verl FSDP workers."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mopd_verl.full_gradient.actor_loss import (
    _actor_micro_batch_loss,
    _actor_micro_batch_token_loss_scores,
    _actor_reverse_kl_advantages,
    _copy_data_proto_rows_to_cpu,
    _data_proto_tensor_device,
    _is_multi_teacher_distill_cfg,
    _labels_from_inputs,
    _response_token_id_matrix_from_inputs,
    _select_by_code_teacher,
    _selected_student_topk_teacher_log_probs,
    _selected_teacher_log_prob_from_inputs,
    _selected_teacher_topk_from_inputs,
    _selected_topk_support_from_inputs,
    _token_contribution_scale,
    _token_mask_contribution_scale,
    _topk_runtime_config,
)
from mopd_verl.full_gradient.config import _cfg_get
from mopd_verl.full_gradient.labels import (
    _TEACHER_LABEL_KEY,
    _labels_from_mapping,
    _non_tensor_list,
    _response_token_count,
    _sample_ids,
    _teacher_labels,
)
from mopd_verl.topk_distill import uses_topk_distill_loss
from verl import DataProto
from verl.utils.device import get_device_id, get_torch_device


_VECTOR_REDUCTION_CHUNK_SIZE = 1 << 26


def _safe_name(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def _all_reduce_sum(value: float) -> float:
    tensor = torch.tensor(float(value), device=get_device_id(), dtype=torch.float64)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor.item())


def _all_reduce_values_sum(values: list[float]) -> list[float]:
    if not values:
        return []
    tensor = torch.tensor(values, device=get_device_id(), dtype=torch.float64)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return [float(value) for value in tensor.tolist()]


def _all_reduce_values_max(values: list[float]) -> list[float]:
    if not values:
        return []
    tensor = torch.tensor(values, device=get_device_id(), dtype=torch.float64)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return [float(value) for value in tensor.tolist()]


def _actor_no_sync_context(actor: Any) -> Any:
    actor_module = getattr(actor, "actor_module", None)
    no_sync = getattr(actor_module, "no_sync", None)
    if callable(no_sync):
        try:
            return no_sync()
        except Exception:
            return nullcontext()
    return nullcontext()


def _finalize_fsdp_after_auxiliary_backward(actor: Any) -> None:
    actor_module = getattr(actor, "actor_module", None)
    if actor_module is None:
        return
    try:
        from torch.distributed.fsdp._common_utils import (
            HandleTrainingState,
            TrainingState,
            _get_module_fsdp_state,
        )
        from torch.distributed.fsdp._runtime_utils import _post_backward_final_callback
    except (AttributeError, ImportError, RuntimeError):
        return

    module_iter = (actor_module,)
    modules = getattr(actor_module, "modules", None)
    if callable(modules):
        try:
            module_iter = tuple(modules())
        except Exception:
            module_iter = (actor_module,)

    state_items: list[tuple[Any, Any]] = []
    seen_states: set[int] = set()
    for module in module_iter:
        try:
            state = _get_module_fsdp_state(module)
        except Exception:
            continue
        if state is None or id(state) in seen_states:
            continue
        seen_states.add(id(state))
        state_items.append((module, state))
    if not state_items:
        return

    root_module, root_state = state_items[0]
    for module, state in state_items:
        if bool(getattr(state, "_is_root", False)):
            root_module, root_state = module, state
            break

    all_states = tuple(getattr(root_state, "_all_fsdp_states", None) or [state for _module, state in state_items])
    backward_states = {HandleTrainingState.BACKWARD_PRE, HandleTrainingState.BACKWARD_POST}
    needs_finalize = any(
        getattr(getattr(state, "_handle", None), "_training_state", None) in backward_states
        for state in all_states
    )
    if not needs_finalize:
        return

    if bool(getattr(root_state, "_is_root", False)):
        try:
            _post_backward_final_callback(root_state, root_module)
            return
        except Exception:
            pass

    for state in all_states:
        try:
            state.training_state = TrainingState.IDLE
        except Exception:
            pass
        handle = getattr(state, "_handle", None)
        if handle is None:
            continue
        try:
            if getattr(handle, "_training_state", None) in backward_states:
                handle._ran_pre_backward_hook = False
                handle._needs_pre_backward_unshard = False
                handle._post_forward_index = None
                handle._training_state = HandleTrainingState.IDLE
                handle._prefetched = False
        except Exception:
            pass
    try:
        root_state._post_backward_callback_queued = False
    except Exception:
        pass


def _all_gather_list(values: list[Any]) -> list[Any]:
    if not torch.distributed.is_initialized():
        return list(values)
    gathered: list[list[Any] | None] = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered, list(values))
    flattened: list[Any] = []
    for part in gathered:
        if part:
            flattened.extend(part)
    return flattened


def _distributed_rank() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
    except (RuntimeError, ValueError):
        return 0
    return 0


def _distributed_world_size() -> int:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_world_size())
    except (RuntimeError, ValueError):
        return 1
    return 1


def _all_ranks_true(value: bool) -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return value
    tensor = torch.tensor(int(value), device=get_device_id(), dtype=torch.int32)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
    return bool(tensor.item())


def _all_ranks_equal_ints(values: list[int]) -> bool:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return True
    local = torch.tensor(values, device=get_device_id(), dtype=torch.int64)
    minimum = local.clone()
    maximum = local.clone()
    torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
    return bool(torch.equal(minimum, maximum))


def _actor_fsdp_size(actor: Any) -> int | None:
    actor_config = getattr(actor, "config", None)
    fsdp_config = _cfg_get(actor_config, "fsdp_config", {})
    raw_fsdp_size = _cfg_get(fsdp_config, "fsdp_size", -1)
    try:
        return int(raw_fsdp_size)
    except (TypeError, ValueError):
        return None


def _gradient_replica_count(actor: Any) -> int:
    world_size = _distributed_world_size()
    if world_size <= 1:
        return 1

    fsdp_size = _actor_fsdp_size(actor)
    if fsdp_size is None:
        return 1

    if fsdp_size <= 0 or fsdp_size >= world_size or world_size % fsdp_size != 0:
        return 1
    return world_size // fsdp_size


def _reduce_gradient_scalars(actor: Any, values: list[float]) -> list[float]:
    return _all_reduce_values_sum(values)


def _actor_has_full_local_params_for_sample_gradient(actor: Any) -> bool:
    return _actor_fsdp_size(actor) == 1


def _chunks_local_sumsq(chunks: tuple[torch.Tensor, ...]) -> float:
    total = 0.0
    for chunk in chunks:
        chunk_sumsq = _chunked_vector_dot(chunk.float(), chunk.float())
        if chunk_sumsq is not None:
            total += chunk_sumsq
    return total


def _chunked_vector_dot(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    if left_flat.numel() == 0 or left_flat.numel() != right_flat.numel():
        return None

    total = 0.0
    for start in range(0, left_flat.numel(), _VECTOR_REDUCTION_CHUNK_SIZE):
        end = min(start + _VECTOR_REDUCTION_CHUNK_SIZE, left_flat.numel())
        left_chunk = left_flat[start:end].float()
        right_chunk = right_flat[start:end].float()
        total += float(torch.dot(left_chunk, right_chunk).item())
    return total


def _safe_cosine(dot: float | None, left_norm: float, right_norm: float) -> float | None:
    if dot is None or left_norm <= 0 or right_norm <= 0:
        return None
    return dot / (left_norm * right_norm)


def _storage_dtype(storage_dtype: str) -> torch.dtype:
    normalized = str(storage_dtype).lower()
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32


def _max_memory_allocated_gb() -> float:
    max_memory_allocated = getattr(get_torch_device(), "max_memory_allocated", None)
    if not callable(max_memory_allocated):
        return 0.0
    try:
        return float(max_memory_allocated()) / (1024**3)
    except (RuntimeError, TypeError, ValueError):
        return 0.0


@dataclass(frozen=True)
class _GradDifferenceSnapshot:
    first_norm_sq: float
    total_norm_sq: float
    first_total_dot: float
    second_norm_sq: float
    first_second_dot: float
    second_total_dot: float
    second_chunks: tuple[torch.Tensor, ...] | None
    second_target_norm_sq: float | None


def _current_grad_scale(actor: Any) -> float:
    scaler = getattr(actor, "scaler", None)
    if scaler is None or not hasattr(scaler, "get_scale"):
        return 1.0
    try:
        scale = float(scaler.get_scale())
    except (TypeError, ValueError):
        return 1.0
    return scale if scale > 0 else 1.0


def _current_grad_cpu_float(parameter: torch.nn.Parameter, scale: float) -> torch.Tensor | None:
    if parameter.grad is None:
        return None
    gradient = parameter.grad.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
    if scale != 1.0:
        gradient = gradient / scale
    return gradient


def _snapshot_current_grad_chunks(
    actor: Any,
    storage_dtype: str,
    *,
    grads_are_scaled: bool | None = None,
) -> tuple[torch.Tensor, ...]:
    dtype = _storage_dtype(storage_dtype)
    if grads_are_scaled is None:
        grads_are_scaled = getattr(actor, "scaler", None) is not None
    scale = _current_grad_scale(actor) if grads_are_scaled else 1.0
    pieces: list[torch.Tensor] = []
    for parameter in _trainable_parameters(actor):
        if parameter.grad is None:
            pieces.append(torch.zeros(parameter.numel(), dtype=dtype, device="cpu"))
            continue
        gradient = parameter.grad.detach()
        if scale != 1.0:
            gradient = gradient / scale
        pieces.append(gradient.reshape(-1).to(device="cpu", dtype=dtype, copy=True))
    return tuple(pieces)


def _current_grad_difference_snapshot(
    actor: Any,
    reference_chunks: tuple[torch.Tensor, ...],
    storage_dtype: str | None = None,
) -> _GradDifferenceSnapshot | None:
    parameters = _trainable_parameters(actor)
    if len(parameters) != len(reference_chunks):
        return None

    dtype = _storage_dtype(storage_dtype) if storage_dtype is not None else None
    second_pieces: list[torch.Tensor] | None = [] if dtype is not None else None
    scale = _current_grad_scale(actor)
    first_sumsq = 0.0
    total_sumsq = 0.0
    first_total_dot = 0.0
    second_sumsq = 0.0
    first_second_dot = 0.0
    second_total_dot = 0.0
    second_target_sumsq = 0.0

    for parameter, first in zip(parameters, reference_chunks):
        if first.numel() != parameter.numel():
            return None
        first_float = first.float()
        total_float = _current_grad_cpu_float(parameter, scale)
        if total_float is None:
            total_float = torch.zeros_like(first_float)
        second_float = total_float - first_float

        first_sumsq += _chunked_vector_dot(first_float, first_float) or 0.0
        total_sumsq += _chunked_vector_dot(total_float, total_float) or 0.0
        first_total_dot += _chunked_vector_dot(first_float, total_float) or 0.0
        second_sumsq += _chunked_vector_dot(second_float, second_float) or 0.0
        first_second_dot += _chunked_vector_dot(first_float, second_float) or 0.0
        second_total_dot += _chunked_vector_dot(second_float, total_float) or 0.0

        if second_pieces is not None and dtype is not None:
            second_piece = second_float.to(dtype=dtype)
            second_pieces.append(second_piece)
            second_piece_float = second_piece.float()
            second_target_sumsq += (
                _chunked_vector_dot(second_piece_float, second_piece_float) or 0.0
            )
            del second_piece_float
        del first_float, total_float, second_float

    scalar_values = [
        first_sumsq,
        total_sumsq,
        first_total_dot,
        second_sumsq,
        first_second_dot,
        second_total_dot,
    ]
    if second_pieces is not None:
        scalar_values.append(second_target_sumsq)
    reduced_values = _reduce_gradient_scalars(actor, scalar_values)

    return _GradDifferenceSnapshot(
        first_norm_sq=reduced_values[0],
        total_norm_sq=reduced_values[1],
        first_total_dot=reduced_values[2],
        second_norm_sq=reduced_values[3],
        first_second_dot=reduced_values[4],
        second_total_dot=reduced_values[5],
        second_chunks=tuple(second_pieces) if second_pieces is not None else None,
        second_target_norm_sq=reduced_values[6] if second_pieces is not None else None,
    )


def _clear_parameter_grads(parameters: tuple[torch.nn.Parameter, ...]) -> None:
    for parameter in parameters:
        parameter.grad = None


def _parameter_grad_dtypes(parameters: tuple[torch.nn.Parameter, ...]) -> tuple[torch.dtype | None, ...]:
    return tuple(parameter.grad.dtype if parameter.grad is not None else None for parameter in parameters)


def _snapshot_parameter_grads(
    parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.Tensor | None, ...]:
    return tuple(
        parameter.grad.detach().to(device="cpu", dtype=torch.float32).clone()
        if parameter.grad is not None
        else None
        for parameter in parameters
    )


def _snapshot_parameter_grads_for_restore(
    parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.Tensor | None, ...]:
    return tuple(
        parameter.grad.detach().to(device="cpu", dtype=parameter.grad.dtype).clone()
        if parameter.grad is not None
        else None
        for parameter in parameters
    )


def _restore_parameter_grads_from_snapshot(
    parameters: tuple[torch.nn.Parameter, ...],
    grad_snapshot: tuple[torch.Tensor | None, ...],
    grad_dtypes: tuple[torch.dtype | None, ...] | None = None,
) -> None:
    for param_idx, parameter in enumerate(parameters):
        if param_idx >= len(grad_snapshot) or grad_snapshot[param_idx] is None:
            parameter.grad = None
            continue
        snapshot = grad_snapshot[param_idx]
        grad_dtype = (
            grad_dtypes[param_idx]
            if grad_dtypes is not None and param_idx < len(grad_dtypes)
            else snapshot.dtype
        )
        try:
            parameter.grad = snapshot.to(device=parameter.device, dtype=grad_dtype).clone()
        except RuntimeError:
            parameter.grad = snapshot.to(device=parameter.device, dtype=parameter.dtype).clone()


def _parameter_grad_snapshot_diff_stats(
    parameters: tuple[torch.nn.Parameter, ...],
    grad_snapshot: tuple[torch.Tensor | None, ...],
) -> dict[str, float]:
    local_diff_sq = 0.0
    local_snapshot_sq = 0.0
    local_max_abs = 0.0
    for param_idx, parameter in enumerate(parameters):
        snapshot = grad_snapshot[param_idx] if param_idx < len(grad_snapshot) else None
        if snapshot is None and parameter.grad is None:
            continue
        if snapshot is None:
            current = parameter.grad.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
            reference = torch.zeros_like(current)
        else:
            reference = snapshot.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
            if parameter.grad is None:
                current = torch.zeros_like(reference)
            else:
                current = parameter.grad.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        if current.numel() != reference.numel():
            return {
                "rel_l2": float("inf"),
                "max_abs": float("inf"),
                "snapshot_norm": float("inf"),
            }
        diff = current - reference
        diff_sq = _chunked_vector_dot(diff, diff)
        snapshot_sq = _chunked_vector_dot(reference, reference)
        if diff_sq is not None:
            local_diff_sq += diff_sq
        if snapshot_sq is not None:
            local_snapshot_sq += snapshot_sq
        if diff.numel() > 0:
            local_max_abs = max(local_max_abs, float(diff.abs().max().item()))
        del current, reference, diff

    diff_sq = max(local_diff_sq, 0.0)
    snapshot_sq = max(local_snapshot_sq, 0.0)
    snapshot_norm = snapshot_sq**0.5
    return {
        "rel_l2": (diff_sq**0.5) / (snapshot_norm + 1e-12),
        "max_abs": local_max_abs,
        "snapshot_norm": snapshot_norm,
    }


def _parameter_grad_target_diff_stats(
    parameters: tuple[torch.nn.Parameter, ...],
    target_map: dict[str, tuple[tuple[torch.Tensor, ...], float]],
) -> dict[str, float]:
    local_diff_sq = 0.0
    local_target_sq = 0.0
    local_max_abs = 0.0
    target_items = list(target_map.values())
    for param_idx, parameter in enumerate(parameters):
        target_total: torch.Tensor | None = None
        for target_chunks, _target_norm_sq in target_items:
            chunk = target_chunks[param_idx].detach().reshape(-1).float()
            target_total = chunk.clone() if target_total is None else target_total.add(chunk)
        if target_total is None:
            continue
        if parameter.grad is None:
            current = torch.zeros_like(target_total)
        else:
            current = parameter.grad.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        if current.numel() != target_total.numel():
            return {
                "rel_l2": float("inf"),
                "max_abs": float("inf"),
                "target_norm": float("inf"),
            }
        diff = current - target_total
        diff_sq = _chunked_vector_dot(diff, diff)
        target_sq = _chunked_vector_dot(target_total, target_total)
        if diff_sq is not None:
            local_diff_sq += diff_sq
        if target_sq is not None:
            local_target_sq += target_sq
        if diff.numel() > 0:
            local_max_abs = max(local_max_abs, float(diff.abs().max().item()))
        del target_total, current, diff

    diff_sq = max(local_diff_sq, 0.0)
    target_sq = max(local_target_sq, 0.0)
    target_norm = target_sq**0.5
    return {
        "rel_l2": (diff_sq**0.5) / (target_norm + 1e-12),
        "max_abs": local_max_abs,
        "target_norm": target_norm,
    }


def _restore_parameter_grads_from_targets(
    parameters: tuple[torch.nn.Parameter, ...],
    target_map: dict[str, tuple[tuple[torch.Tensor, ...], float]],
    grad_dtypes: tuple[torch.dtype | None, ...] | None = None,
) -> None:
    target_items = list(target_map.values())
    for param_idx, parameter in enumerate(parameters):
        total_chunk: torch.Tensor | None = None
        for target_chunks, _target_norm_sq in target_items:
            chunk = target_chunks[param_idx].detach().reshape(-1).float()
            total_chunk = chunk.clone() if total_chunk is None else total_chunk.add(chunk)
        if total_chunk is None:
            parameter.grad = None
            continue
        restore_dtype = parameter.dtype
        if grad_dtypes is not None and param_idx < len(grad_dtypes):
            original_grad_dtype = grad_dtypes[param_idx]
            if original_grad_dtype is not None:
                restore_dtype = original_grad_dtype
        restored = total_chunk.reshape(parameter.shape).to(device=parameter.device, dtype=restore_dtype)
        try:
            parameter.grad = restored.clone()
        except RuntimeError:
            parameter.grad = total_chunk.reshape(parameter.shape).to(
                device=parameter.device,
                dtype=parameter.dtype,
            ).clone()


def _finite_values(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _std(values: list[float]) -> float | None:
    return float(np.std(values)) if values else None


def _json_safe(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_jsonl_rows(output_dir: str | None, filename: str, rows: list[dict[str, Any]]) -> None:
    if not output_dir or not rows:
        return
    path = Path(str(output_dir)) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")


def _trainable_parameters(actor: Any) -> tuple[torch.nn.Parameter, ...]:
    optimizer = getattr(actor, "actor_optimizer", None)
    if optimizer is not None:
        params = []
        seen: set[int] = set()
        for group in getattr(optimizer, "param_groups", []):
            for parameter in group.get("params", []):
                if parameter.requires_grad and id(parameter) not in seen:
                    params.append(parameter)
                    seen.add(id(parameter))
        if params:
            return tuple(params)
    return tuple(parameter for parameter in actor.actor_module.parameters() if parameter.requires_grad)


def _gradient_chunk_pair_stats(
    actor: Any,
    reference_chunks: tuple[torch.Tensor, ...],
    candidate_chunks: tuple[torch.Tensor, ...],
    parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if len(reference_chunks) != len(candidate_chunks) or len(reference_chunks) != len(parameters):
        return {"shape_mismatch": 1.0}, []

    local_ref_sq = 0.0
    local_candidate_sq = 0.0
    local_diff_sq = 0.0
    local_dot = 0.0
    local_max_abs = 0.0

    for reference, candidate, parameter in zip(reference_chunks, candidate_chunks, parameters):
        if reference.numel() != candidate.numel() or reference.numel() != parameter.numel():
            return {"shape_mismatch": 1.0}, []
        reference_float = reference.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        candidate_float = candidate.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        diff = candidate_float - reference_float
        ref_sq = _chunked_vector_dot(reference_float, reference_float) or 0.0
        candidate_sq = _chunked_vector_dot(candidate_float, candidate_float) or 0.0
        diff_sq = _chunked_vector_dot(diff, diff) or 0.0
        dot = _chunked_vector_dot(candidate_float, reference_float) or 0.0
        max_abs = float(diff.abs().max().item()) if diff.numel() > 0 else 0.0

        local_ref_sq += ref_sq
        local_candidate_sq += candidate_sq
        local_diff_sq += diff_sq
        local_dot += dot
        local_max_abs = max(local_max_abs, max_abs)
        del reference_float, candidate_float, diff

    ref_sq, candidate_sq, diff_sq, dot = _reduce_gradient_scalars(
        actor,
        [local_ref_sq, local_candidate_sq, local_diff_sq, local_dot],
    )
    max_abs = _all_reduce_values_max([local_max_abs])[0]
    ref_norm = max(ref_sq, 0.0) ** 0.5
    candidate_norm = max(candidate_sq, 0.0) ** 0.5
    diff_norm = max(diff_sq, 0.0) ** 0.5
    stats = {
        "reference_norm": ref_norm,
        "candidate_norm": candidate_norm,
        "diff_norm": diff_norm,
        "rel_l2": diff_norm / (ref_norm + 1e-12),
        "norm_ratio": candidate_norm / (ref_norm + 1e-12),
        "max_abs": max_abs,
    }
    cosine = _safe_cosine(dot, candidate_norm, ref_norm)
    if cosine is not None:
        stats["cosine"] = cosine
    if ref_sq > 0.0:
        stats["projection_share"] = dot / ref_sq

    return stats, []


class SequentialBackwardDomainGradientTracker:
    """Track domain and sample gradient geometry from the real actor backward pass."""

    def __init__(self, actor: Any, cfg: dict[str, Any]):
        self.actor = actor
        self.cfg = cfg
        self.domains = list(cfg.get("domains", []))
        self.storage_dtype = str(cfg.get("storage_dtype", "float32"))
        self.step = int(cfg.get("step", 0) or 0)
        self.execution_timing = str(cfg.get("execution_timing", "pre_update")).lower()
        self.pre_update_audit = self.execution_timing in {"pre_update", "pre-audit", "pre_audit"}
        self.domain_gradient_enabled = bool(cfg.get("domain_gradient_enabled", cfg.get("enabled", False)))
        self.full_grad_training_parity_freq_steps = int(
            cfg.get("full_grad_training_parity_freq_steps", 1) or 1
        )
        self.full_gradient_direct_recompute_enabled = self.domain_gradient_enabled and bool(
            cfg.get("full_gradient_direct_recompute_enabled", True)
        )
        self.sequence_masked_target_enabled = self.domain_gradient_enabled and bool(
            cfg.get("sequence_masked_target_enabled", False)
        )
        self.sequence_masked_target_use_as_primary = self.sequence_masked_target_enabled and bool(
            cfg.get("sequence_masked_target_use_as_primary", False)
        )
        self.domain_direct_recompute_closure_rel_l2_threshold = float(
            cfg.get("domain_direct_recompute_closure_rel_l2_threshold", 0.02)
        )
        self.inject_opd_teacher_from_domain_partition = bool(
            cfg.get("inject_opd_teacher_from_domain_partition", False)
        )
        self.sample_gradient_enabled = bool(cfg.get("sample_gradient_enabled", False))
        self._distributed_world_size = _distributed_world_size()
        requested_sample_norm = self.sample_gradient_enabled and bool(cfg.get("sample_gradient_norm_enabled", True))
        requested_sample_cos = self.sample_gradient_enabled and bool(cfg.get("sample_gradient_cos_enabled", False))
        requested_token_gradient = bool(cfg.get("token_gradient_enabled", False))
        self._sample_gradient_uses_full_local_params = _actor_has_full_local_params_for_sample_gradient(actor)
        self._token_gradient_sequence_replay_supported = (
            self.sequence_masked_target_enabled
            and self.sequence_masked_target_use_as_primary
        )
        self._sample_gradient_norm_distributed_unsupported = (
            requested_sample_norm and not self._sample_gradient_uses_full_local_params
        )
        self._sample_gradient_cos_distributed_unsupported = (
            requested_sample_cos and not self._sample_gradient_uses_full_local_params
        )
        self._token_gradient_distributed_unsupported = (
            requested_token_gradient
            and not self._sample_gradient_uses_full_local_params
            and not self._token_gradient_sequence_replay_supported
        )
        self.sample_norm_enabled = requested_sample_norm and not self._sample_gradient_norm_distributed_unsupported
        self.sample_cos_enabled = requested_sample_cos and not self._sample_gradient_cos_distributed_unsupported
        self.token_gradient_enabled = requested_token_gradient and not self._token_gradient_distributed_unsupported
        self.token_gradient_top_k = max(
            1,
            int(cfg.get("token_gradient_top_k", 100) or 100),
        )
        self.token_gradient_gap_selection_enabled = bool(
            cfg.get("token_gradient_gap_selection_enabled", True)
        )
        self.token_gradient_gap_abs_selection_enabled = bool(
            cfg.get("token_gradient_gap_abs_selection_enabled", True)
        )
        self.token_gradient_loss_abs_selection_enabled = bool(
            cfg.get("token_gradient_loss_abs_selection_enabled", True)
        )
        token_gradient_top_p = cfg.get("token_gradient_top_p", 0.10)
        self.token_gradient_top_p = min(
            1.0,
            max(0.0, float(0.10 if token_gradient_top_p is None else token_gradient_top_p)),
        )
        self.token_gradient_strict_grad_restore = self.token_gradient_enabled and bool(
            cfg.get("token_gradient_strict_grad_restore", False)
        )
        self.sample_gradient_backward_recompute_enabled = self.sample_cos_enabled and bool(
            cfg.get("sample_gradient_backward_recompute_enabled", True)
        )
        self.sample_gradient_backward_sync_enabled = self.sample_gradient_backward_recompute_enabled and bool(
            cfg.get("sample_gradient_backward_sync_enabled", True)
        )
        self.token_gradient_backward_recompute_enabled = self.token_gradient_enabled and bool(
            cfg.get("token_gradient_backward_recompute_enabled", True)
        )
        self.token_gradient_backward_sync_enabled = self.token_gradient_backward_recompute_enabled and bool(
            cfg.get("token_gradient_backward_sync_enabled", True)
        )
        self._sample_gradient_distributed_unsupported = not (
            self.sample_norm_enabled
            or self.sample_cos_enabled
            or self.token_gradient_enabled
            or not (requested_sample_norm or requested_sample_cos or requested_token_gradient)
        )
        self.sample_log_sample_level = (
            bool(cfg.get("sample_gradient_log_sample_level", True))
            and self.sample_norm_enabled
        )
        self.output_dir = str(cfg.get("output_dir", ""))
        self._sample_counts: dict[str, int] = {}
        self._first_domain_chunks: tuple[torch.Tensor, ...] | None = None
        self._expected_first_domain_samples: int | None = None
        self._started_at = 0.0
        self._domain_partition_meta = cfg.get("domain_partition", {})
        self._prepared_supported = len(self.domains) in (1, 2)
        self._domain_partition_injected_domain = 0.0
        self._domain_partition_injected_opd_teacher = 0.0
        self._sample_records: list[dict[str, Any]] = []
        self._sample_candidates: dict[str, list[dict[str, Any]]] = {}
        self._domain_recompute_candidates: dict[str, list[dict[str, Any]]] = {}
        self._schedule_candidates: list[dict[str, Any]] = []
        self._token_gradient_candidates: dict[str, list[dict[str, Any]]] = {}
        self._token_gradient_selected_sample_ids: dict[str, set[str]] = {}
        self._micro_batch_index = 0
        self._sample_zero_norm_count = 0
        self._last_audit_total_chunks: tuple[torch.Tensor, ...] = tuple()
        self._last_sequence_total_chunks: tuple[torch.Tensor, ...] = tuple()
        self._use_dynamic_micro_batch = False

    def _should_log_full_grad_training_parity(self) -> bool:
        freq_steps = int(self.full_grad_training_parity_freq_steps)
        return freq_steps >= 0 and self.step % max(1, freq_steps) == 0

    def _domain_target_storage_dtype(self) -> str:
        return self.storage_dtype

    def _domain_direct_recompute_active(self) -> bool:
        return bool(
            self.domain_gradient_enabled
            and self.full_gradient_direct_recompute_enabled
            and self._prepared_supported
            and self.domains
        )

    def prepare_micro_batches(
        self,
        micro_batches: list[Any],
        *,
        batch_idx_list: list[list[int]] | None = None,
    ) -> list[tuple[str | None, Any]]:
        original_micro_batches = list(micro_batches)
        self._expected_first_domain_samples = None
        self._domain_partition_injected_domain = 0.0
        self._domain_partition_injected_opd_teacher = 0.0
        partition_meta = self._domain_partition_meta if isinstance(self._domain_partition_meta, dict) else {}
        self._inject_partition_labels(
            original_micro_batches,
            partition_meta,
            batch_idx_list=batch_idx_list,
        )
        partition_aligned = (
            bool(partition_meta.get("aligned", False))
            if partition_meta
            else self._distributed_world_size <= 1
        )
        locally_supported = len(self.domains) in (1, 2) and partition_aligned
        buckets: dict[str, list[tuple[str, Any]]] = {domain: [] for domain in self.domains}
        if locally_supported:
            for micro_batch in original_micro_batches:
                labels = _teacher_labels(micro_batch)
                unique_labels = set(labels)
                if len(unique_labels) != 1:
                    locally_supported = False
                    break
                domain = next(iter(unique_labels))
                if domain not in buckets:
                    locally_supported = False
                    break
                buckets[domain].append((domain, micro_batch))
        if locally_supported:
            locally_supported = all(buckets[domain] for domain in self.domains)

        globally_supported = _all_ranks_true(locally_supported)
        domain_sample_counts = [
            sum(len(micro_batch) for _, micro_batch in buckets[domain]) if locally_supported else 0
            for domain in self.domains
        ]
        domain_micro_batch_counts = [
            len(buckets[domain]) if locally_supported else 0 for domain in self.domains
        ]
        counts_aligned = False
        if globally_supported:
            micro_batch_counts_aligned = _all_ranks_equal_ints(domain_micro_batch_counts)
            sample_counts_aligned = _all_ranks_equal_ints(domain_sample_counts)
            meta_sample_counts = partition_meta.get("domain_block_sample_counts", {})
            meta_counts_match = all(
                int(meta_sample_counts.get(domain, -1)) == domain_sample_counts[idx]
                for idx, domain in enumerate(self.domains)
            ) if meta_sample_counts else self._distributed_world_size <= 1
            counts_aligned = micro_batch_counts_aligned and sample_counts_aligned and meta_counts_match
        self._prepared_supported = globally_supported and counts_aligned
        if self.domain_gradient_enabled and not self._prepared_supported:
            self.domain_gradient_enabled = False
        if not self._prepared_supported:
            self._expected_first_domain_samples = None
            self._first_domain_chunks = None
            return [(None, item) for item in original_micro_batches]

        if partition_meta.get("domain_order") not in (None, list(self.domains)):
            self.domain_gradient_enabled = False
            self._prepared_supported = False
            return [(None, item) for item in original_micro_batches]

        self._expected_first_domain_samples = domain_sample_counts[0]
        ordered: list[tuple[str | None, Any]] = []
        for domain in self.domains:
            ordered.extend(buckets[domain])
        return ordered

    def _inject_partition_labels(
        self,
        micro_batches: list[Any],
        partition_meta: dict[str, Any],
        *,
        batch_idx_list: list[list[int]] | None,
    ) -> None:
        if not partition_meta or not bool(partition_meta.get("aligned", False)):
            return
        if not batch_idx_list or len(batch_idx_list) != len(micro_batches):
            return
        domains = list(partition_meta.get("domain_order") or self.domains)
        counts_by_domain = partition_meta.get("domain_block_sample_counts", {})
        if not domains or not isinstance(counts_by_domain, dict):
            return
        boundaries: list[tuple[int, int, str]] = []
        start = 0
        for domain in domains:
            count = int(counts_by_domain.get(domain, 0) or 0)
            if count <= 0:
                continue
            end = start + count
            boundaries.append((start, end, str(domain)))
            start = end
        if not boundaries:
            return
        for micro_batch, indices in zip(micro_batches, batch_idx_list):
            labels: list[str] = []
            for index in indices:
                label = None
                idx = int(index)
                for start, end, domain in boundaries:
                    if start <= idx < end:
                        label = domain
                        break
                labels.append(label or "unknown")
            if not labels or all(label == "unknown" for label in labels):
                continue
            current_labels = _teacher_labels(micro_batch)
            if any(label != "unknown" for label in current_labels):
                continue
            label_array = np.array(labels, dtype=object)
            micro_batch.non_tensor_batch["domain"] = label_array.copy()
            self._domain_partition_injected_domain = 1.0
            if self.inject_opd_teacher_from_domain_partition:
                micro_batch.non_tensor_batch[_TEACHER_LABEL_KEY] = label_array.copy()
                self._domain_partition_injected_opd_teacher = 1.0

    def start_mini_batch(self) -> None:
        self._sample_counts = {}
        self._first_domain_chunks = None
        self._started_at = time.perf_counter()
        self._clear_mini_batch_cpu_refs()
        self._micro_batch_index = 0
        self._sample_zero_norm_count = 0
        self._last_audit_total_chunks = tuple()
        self._last_sequence_total_chunks = tuple()
        self._use_dynamic_micro_batch = False

    def _clear_mini_batch_cpu_refs(self) -> None:
        self._sample_records = []
        self._sample_candidates = {}
        self._domain_recompute_candidates = {}
        self._schedule_candidates = []
        self._token_gradient_candidates = {}
        self._token_gradient_selected_sample_ids = {}

    def run_pre_update_audit(
        self,
        tracked_micro_batches: list[tuple[str | None, Any]],
        *,
        on_policy: bool,
        use_dynamic_micro_batch: bool,
        ppo_mini_batch_size: int,
        gradient_accumulation: int,
    ) -> dict[str, float]:
        self.start_mini_batch()
        self._use_dynamic_micro_batch = bool(use_dynamic_micro_batch)
        metrics: dict[str, float] = {
            "global/audit/pre_update_audit_used": 1.0,
        }
        try:
            for domain, micro_batch in tracked_micro_batches:
                if use_dynamic_micro_batch:
                    response_mask = micro_batch.batch["response_mask"]
                    loss_scale_factor = response_mask.shape[0] / max(ppo_mini_batch_size, 1)
                else:
                    loss_scale_factor = 1.0 / max(gradient_accumulation, 1)
                self.record_pre_update_micro_batch(
                    domain,
                    micro_batch,
                    loss_scale_factor=float(loss_scale_factor),
                    on_policy=on_policy,
                )
            metrics.update(self.finish_mini_batch())
        finally:
            _finalize_fsdp_after_auxiliary_backward(self.actor)
            _clear_parameter_grads(_trainable_parameters(self.actor))
        return metrics

    def record_pre_update_micro_batch(
        self,
        domain: str | None,
        micro_batch: Any,
        *,
        loss_scale_factor: float,
        on_policy: bool,
    ) -> None:
        if not (
            self.sample_norm_enabled
            or self.sample_cos_enabled
            or self.token_gradient_enabled
            or self._domain_direct_recompute_active()
        ):
            self._micro_batch_index += 1
            return
        labels = _teacher_labels(micro_batch)
        resolved_domain = domain
        if resolved_domain is None and len(set(labels)) == 1:
            resolved_domain = labels[0]
        rank = _distributed_rank()
        fallback_prefix = f"step{self.step}:rank{rank}:micro{self._micro_batch_index}"
        sample_ids = _sample_ids(
            micro_batch,
            self.step,
            fallback_prefix=fallback_prefix,
        )
        sample_id = sample_ids[0] if sample_ids else fallback_prefix
        context: dict[str, Any] = {
            "step": self.step,
            "domain": resolved_domain or "unknown",
            "sample_id": sample_id,
            "sample_count": len(micro_batch),
            "is_true_sample_level": len(micro_batch) == 1,
            "micro_batch_index": self._micro_batch_index,
            "effective_tokens": _response_token_count(micro_batch),
            "loss_scale_factor": float(loss_scale_factor),
            "on_policy": bool(on_policy),
        }
        row: dict[str, Any] = {
            **context,
            "computed_for_cos": False,
            "sample_grad_norm": None,
            "sample_to_domain_cos": None,
            "sample_projection_share": None,
            "sample_projection_share_normalized": None,
        }
        if self.sample_norm_enabled or self.sample_cos_enabled:
            self._sample_records.append(row)

        domain_name = str(context["domain"])
        stored_micro_batch = _copy_data_proto_rows_to_cpu(
            micro_batch,
            list(range(len(micro_batch))),
        )
        if stored_micro_batch is not None:
            self._schedule_candidates.append(
                {
                    "context": dict(context),
                    "domain": domain_name,
                    "micro_batch": stored_micro_batch,
                    "micro_batch_index": int(context["micro_batch_index"]),
                    "loss_scale_factor": float(loss_scale_factor),
                    "on_policy": bool(on_policy),
                }
            )
        if stored_micro_batch is not None and self._domain_direct_recompute_active() and domain_name in self.domains:
            self._domain_recompute_candidates.setdefault(domain_name, []).append(
                {
                    "context": dict(context),
                    "micro_batch": stored_micro_batch,
                }
            )
        if stored_micro_batch is not None and self.sample_cos_enabled:
            self._sample_candidates.setdefault(domain_name, []).append(
                {"row": row, "micro_batch": stored_micro_batch}
            )
        if stored_micro_batch is not None and self.token_gradient_enabled:
            token_domains = [domain_name] if domain_name in self.domains else []
            if not token_domains:
                label_set = set(labels)
                token_domains = [candidate for candidate in self.domains if candidate in label_set]
            for token_domain in token_domains:
                self._store_token_gradient_candidates(
                    token_domain,
                    micro_batch,
                    row,
                    stored_micro_batch=stored_micro_batch,
                )
        if self._prepared_supported and domain_name in self.domains:
            self._sample_counts[domain_name] = self._sample_counts.get(domain_name, 0) + len(micro_batch)
        self._micro_batch_index += 1

    def finish_mini_batch(self) -> dict[str, float]:
        try:
            return self._finish_mini_batch_impl()
        finally:
            self._clear_mini_batch_cpu_refs()

    def _finish_mini_batch_impl(self) -> dict[str, float]:
        finish_started_at = time.perf_counter()
        metrics: dict[str, float] = {
            "global/audit/full_gradient_autograd_unavailable": 0.0,
            "global/audit/full_gradient_true_backward_fallback": 0.0,
            "global/audit/sample_gradient_zero_norm_count": 0.0,
            "global/audit/full_gradient_domain_sequential_available": 0.0,
            "global/audit/full_gradient_domain_sequential_unsupported": float(not self._prepared_supported),
            "global/audit/domain_partition_injected_domain": self._domain_partition_injected_domain,
            "global/audit/domain_partition_injected_opd_teacher": self._domain_partition_injected_opd_teacher,
            "global/audit/full_gradient_domain_direct_recompute_closure_rel_l2_threshold": (
                self.domain_direct_recompute_closure_rel_l2_threshold
            ),
            "global/audit/full_gradient_execution_timing_pre_update": float(self.pre_update_audit),
            "global/full_grad_cost/backward_seconds": time.perf_counter() - self._started_at,
            "global/full_grad_cost/max_memory_allocated_gb": _max_memory_allocated_gb(),
        }
        replica_count = _gradient_replica_count(self.actor)
        if replica_count > 1:
            metrics["global/audit/full_gradient_replicated_all_reduce"] = 1.0
            metrics["global/audit/full_gradient_replica_count"] = float(replica_count)
        if self._sample_gradient_distributed_unsupported:
            metrics["global/audit/sample_gradient_distributed_unsupported"] = 1.0
            metrics["global/audit/sample_gradient_distributed_world_size"] = float(self._distributed_world_size)
        if self._sample_gradient_norm_distributed_unsupported:
            metrics["global/audit/sample_gradient_norm_distributed_unsupported"] = 1.0
        if self._sample_gradient_cos_distributed_unsupported:
            metrics["global/audit/sample_gradient_cos_distributed_unsupported"] = 1.0
        if self._token_gradient_distributed_unsupported:
            metrics["global/audit/token_gradient_distributed_unsupported"] = 1.0
        if (
            self.token_gradient_enabled
            and not self._sample_gradient_uses_full_local_params
            and self._token_gradient_sequence_replay_supported
        ):
            metrics["global/audit/token_gradient_distributed_sequence_replay_enabled"] = 1.0
        first_chunks = self._first_domain_chunks
        self._first_domain_chunks = None
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]] = {}
        direct_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]] = {}
        domain_target_source = 0.0
        keep_parity_chunks = self.pre_update_audit and self._should_log_full_grad_training_parity()

        can_direct_domain_recompute = self._domain_direct_recompute_active()
        if (
            self.domain_gradient_enabled
            and self._prepared_supported
            and len(self.domains) == 1
            and (first_chunks is not None or can_direct_domain_recompute)
        ):
            direct_metrics, direct_targets = self._recompute_direct_domain_targets()
            metrics.update(direct_metrics)
            if self._direct_domain_targets_pass_closure_gate(direct_targets, direct_metrics):
                domain_metrics, domain_targets = self._finish_direct_domain_gradient_metrics(direct_targets)
                domain_target_source = 1.0
            elif first_chunks is not None:
                if direct_targets:
                    metrics["global/audit/full_gradient_domain_direct_recompute_rejected_by_closure"] = 1.0
                domain_metrics, domain_targets = self._finish_single_domain_gradient_metrics(first_chunks)
                domain_target_source = 3.0
            else:
                if direct_targets:
                    metrics["global/audit/full_gradient_domain_direct_recompute_rejected_by_closure"] = 1.0
                domain_metrics, domain_targets = {}, {}
            metrics.update(domain_metrics)
        elif (
            self.domain_gradient_enabled
            and self._prepared_supported
            and len(self.domains) == 2
            and (first_chunks is not None or can_direct_domain_recompute)
        ):
            domain_summary_started_at = time.perf_counter()
            direct_metrics, direct_targets = self._recompute_direct_domain_targets()
            metrics.update(direct_metrics)
            if self._direct_domain_targets_pass_closure_gate(direct_targets, direct_metrics):
                domain_metrics, domain_targets = self._finish_direct_domain_gradient_metrics(direct_targets)
                domain_target_source = 1.0
            elif first_chunks is not None:
                if direct_targets:
                    metrics["global/audit/full_gradient_domain_direct_recompute_rejected_by_closure"] = 1.0
                domain_metrics, domain_targets = self._finish_domain_gradient_metrics(first_chunks)
                domain_target_source = 2.0
            else:
                if direct_targets:
                    metrics["global/audit/full_gradient_domain_direct_recompute_rejected_by_closure"] = 1.0
                domain_metrics, domain_targets = {}, {}
            metrics["global/full_grad_cost/domain_summary_seconds"] = (
                time.perf_counter() - domain_summary_started_at
            )
            metrics.update(domain_metrics)
        sequence_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]] = {}
        sequence_total_chunks: tuple[torch.Tensor, ...] = tuple()
        if self.domain_gradient_enabled and self._prepared_supported and self.sequence_masked_target_enabled:
            (
                sequence_metrics,
                sequence_targets,
                sequence_total_chunks,
                _sequence_total_norm_sq,
            ) = self._recompute_sequence_domain_targets()
            metrics.update(sequence_metrics)
            if keep_parity_chunks and sequence_total_chunks:
                self._last_sequence_total_chunks = sequence_total_chunks
            if self.sequence_masked_target_use_as_primary and sequence_targets:
                sequence_domain_metrics, sequence_domain_targets = self._finish_direct_domain_gradient_metrics(
                    sequence_targets
                )
                metrics.update(sequence_domain_metrics)
                domain_targets = sequence_domain_targets
                domain_target_source = 4.0
                metrics["global/audit/full_gradient_domain_sequence_masked_primary_used"] = 1.0
                metrics["global/audit/full_gradient_domain_sequence_masked_replay_used"] = 1.0
                metrics["global/audit/full_gradient_domain_direct_recompute_used"] = 0.0
        if not domain_targets:
            domain_target_source = 0.0
        metrics["global/audit/full_gradient_domain_target_source"] = domain_target_source
        metrics["global/audit/full_gradient_domain_target_source_sequence_masked_replay"] = float(
            domain_target_source == 4.0
        )
        metrics["global/audit/full_gradient_domain_target_trusted"] = float(domain_target_source == 4.0)

        if domain_targets:
            _finalize_fsdp_after_auxiliary_backward(self.actor)
            chosen_reference_chunks = (
                self._summed_domain_target_reference_chunks(domain_targets)
                if self.pre_update_audit
                else _snapshot_current_grad_chunks(
                    self.actor,
                    "float32",
                    grads_are_scaled=True,
                )
            )
            metrics.update(
                self._domain_target_closure_metrics(
                    domain_targets,
                    reference_chunks=chosen_reference_chunks,
                    prefix="global/full_grad_closure/chosen_target",
                )
            )
            if keep_parity_chunks:
                self._last_audit_total_chunks = chosen_reference_chunks
            else:
                del chosen_reference_chunks

        if self.sample_cos_enabled and domain_targets:
            metrics.update(self._sample_cos_metrics(domain_targets))
        metrics.update(self._sample_norm_metrics())
        if self.token_gradient_enabled:
            metrics.update(self._token_gradient_metrics(domain_targets))
        metrics["global/audit/sample_gradient_zero_norm_count"] = _all_reduce_sum(
            self._sample_zero_norm_count
        )
        if self.sample_log_sample_level:
            _write_jsonl_rows(self.output_dir, "sample_grad_metrics.jsonl", self._sample_records)
        metrics["global/full_grad_cost/finish_mini_batch_seconds"] = time.perf_counter() - finish_started_at
        return metrics

    def full_grad_training_parity_metrics(self) -> dict[str, float]:
        audit_total_chunks = self._last_audit_total_chunks
        sequence_total_chunks = self._last_sequence_total_chunks
        self._last_audit_total_chunks = tuple()
        self._last_sequence_total_chunks = tuple()
        if not self._should_log_full_grad_training_parity():
            return {}
        if not audit_total_chunks and not sequence_total_chunks:
            return {}

        parameters = _trainable_parameters(self.actor)
        training_chunks = _snapshot_current_grad_chunks(
            self.actor,
            "float32",
            grads_are_scaled=True,
        )
        if len(training_chunks) != len(parameters):
            return {}

        metrics: dict[str, float] = {}
        comparisons = (
            (
                "global/full_grad_training_parity/audit_total_vs_training_total",
                audit_total_chunks,
                training_chunks,
            ),
            (
                "global/full_grad_training_parity/sequence_total_vs_training_total",
                training_chunks,
                sequence_total_chunks,
            ),
        )
        for prefix, reference_chunks, candidate_chunks in comparisons:
            if not reference_chunks or not candidate_chunks:
                continue
            if len(reference_chunks) != len(parameters) or len(candidate_chunks) != len(parameters):
                continue
            stats, _ = _gradient_chunk_pair_stats(
                self.actor,
                reference_chunks,
                candidate_chunks,
                parameters,
            )
            if stats.get("shape_mismatch"):
                continue
            for key, value in stats.items():
                metrics[f"{prefix}/{key}"] = float(value)
        return metrics

    def _store_token_gradient_candidates(
        self,
        domain: str,
        micro_batch: DataProto,
        context: dict[str, Any],
        *,
        stored_micro_batch: DataProto | None = None,
    ) -> None:
        token_candidates = self._select_token_gradient_candidates(
            micro_batch,
            domain=domain,
            fallback_prefix=str(context.get("sample_id", f"step{self.step}")),
            on_policy=bool(context.get("on_policy", False)),
            loss_scale_factor=float(context.get("loss_scale_factor", 1.0) or 1.0),
        )
        if not token_candidates:
            return

        if stored_micro_batch is not None:
            normalized_rows = []
            for row in token_candidates:
                normalized_row = dict(row)
                normalized_row["original_sample_index"] = int(row["sample_index"])
                normalized_row["source_micro_batch_index"] = int(context.get("micro_batch_index", 0) or 0)
                normalized_rows.append(normalized_row)
            self._token_gradient_candidates.setdefault(domain, []).append(
                {
                    "context": dict(context),
                    "micro_batch": stored_micro_batch,
                    "tokens": normalized_rows,
                }
            )
            return

        sample_indices = sorted({int(row["sample_index"]) for row in token_candidates})
        if not sample_indices:
            return
        sample_micro_batch = _copy_data_proto_rows_to_cpu(micro_batch, sample_indices)
        if sample_micro_batch is None:
            return
        index_map = {sample_idx: new_idx for new_idx, sample_idx in enumerate(sample_indices)}
        normalized_rows = []
        for row in token_candidates:
            original_sample_index = int(row["sample_index"])
            normalized_row = dict(row)
            normalized_row["original_sample_index"] = original_sample_index
            normalized_row["sample_index"] = index_map[original_sample_index]
            normalized_row["source_micro_batch_index"] = int(context.get("micro_batch_index", 0) or 0)
            normalized_rows.append(normalized_row)
        self._token_gradient_candidates.setdefault(domain, []).append(
            {
                "context": dict(context),
                "micro_batch": sample_micro_batch,
                "tokens": normalized_rows,
            }
        )

    def _direct_domain_targets_pass_closure_gate(
        self,
        direct_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]],
        direct_metrics: dict[str, float],
    ) -> bool:
        if not direct_targets or not all(domain in direct_targets for domain in self.domains):
            return False
        if self.sequence_masked_target_enabled:
            return False
        if self.pre_update_audit:
            return False
        rel_l2 = direct_metrics.get("global/full_grad_closure/domain_sum_vs_training/rel_l2")
        if rel_l2 is None:
            return False
        try:
            rel_l2_float = float(rel_l2)
        except (TypeError, ValueError):
            return False
        return rel_l2_float <= self.domain_direct_recompute_closure_rel_l2_threshold

    def _recompute_direct_domain_targets(
        self,
    ) -> tuple[dict[str, float], dict[str, tuple[tuple[torch.Tensor, ...], float]]]:
        metrics: dict[str, float] = {}
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]] = {}
        if not self._domain_direct_recompute_active():
            return metrics, domain_targets
        if not any(self._domain_recompute_candidates.get(domain) for domain in self.domains):
            return metrics, domain_targets

        started_at = time.perf_counter()
        parameters = _trainable_parameters(self.actor)
        if not parameters:
            return metrics, domain_targets
        _finalize_fsdp_after_auxiliary_backward(self.actor)
        grad_dtypes = _parameter_grad_dtypes(parameters)
        grad_snapshot = _snapshot_parameter_grads_for_restore(parameters)
        training_total_chunks = (
            tuple()
            if self.pre_update_audit
            else _snapshot_current_grad_chunks(
                self.actor,
                "float32",
                grads_are_scaled=True,
            )
        )
        storage_dtype = self._domain_target_storage_dtype()
        try:
            for domain in self.domains:
                candidates = self._domain_recompute_candidates.get(domain, [])
                if not candidates:
                    continue
                _clear_parameter_grads(parameters)
                for candidate in candidates:
                    micro_batch = candidate["micro_batch"]
                    context = candidate["context"]
                    loss = _actor_micro_batch_loss(
                        self.actor,
                        micro_batch,
                        loss_scale_factor=float(context.get("loss_scale_factor", 1.0) or 1.0),
                        on_policy=bool(context.get("on_policy", False)),
                        safe_logprob_backward=False,
                    )
                    loss.backward()
                    del loss
                    try:
                        micro_batch.to("cpu")
                    except Exception:
                        pass
                _finalize_fsdp_after_auxiliary_backward(self.actor)
                chunks = _snapshot_current_grad_chunks(
                    self.actor,
                    storage_dtype,
                    grads_are_scaled=False,
                )
                local_norm_sq = _chunks_local_sumsq(chunks)
                norm_sq = _reduce_gradient_scalars(self.actor, [local_norm_sq])[0]
                if norm_sq > 0.0:
                    domain_targets[domain] = (chunks, norm_sq)
        except Exception:
            metrics["global/audit/full_gradient_domain_direct_recompute_error"] = 1.0
            domain_targets = {}
        finally:
            _finalize_fsdp_after_auxiliary_backward(self.actor)
            _restore_parameter_grads_from_snapshot(parameters, grad_snapshot, grad_dtypes=grad_dtypes)

        if domain_targets:
            metrics["global/audit/full_gradient_domain_direct_recompute_available"] = 1.0
            if self.pre_update_audit:
                metrics["global/audit/full_gradient_domain_direct_recompute_pre_update"] = 1.0
            metrics["global/full_grad_cost/domain_direct_recompute_seconds"] = (
                time.perf_counter() - started_at
            )
            if training_total_chunks:
                metrics.update(
                    self._domain_target_closure_metrics(
                        domain_targets,
                        reference_chunks=training_total_chunks,
                    )
                )
        return metrics, domain_targets

    def _build_sequence_target_mask(
        self,
        *,
        target_spec: dict[str, Any],
        slot: dict[str, Any],
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_type = str(target_spec.get("type", ""))
        if target_type == "total":
            return response_mask.detach().float().clone()
        if target_type == "domain":
            target_domain = str(target_spec.get("domain", ""))
            if str(slot.get("domain", "")) == target_domain:
                return response_mask.detach().float().clone()
            return torch.zeros_like(response_mask, dtype=torch.float32)
        if target_type == "sample":
            mask = torch.zeros_like(response_mask, dtype=torch.float32)
            if _distributed_rank() != int(target_spec.get("owner_rank", -1)):
                return mask
            if int(slot.get("micro_batch_index", -1)) != int(
                target_spec.get("source_micro_batch_index", -2)
            ):
                return mask
            sample_idx = int(target_spec.get("sample_index", 0))
            if 0 <= sample_idx < int(response_mask.shape[0]):
                mask[sample_idx] = response_mask.detach().float()[sample_idx]
            return mask
        if target_type == "token_selection":
            mask = torch.zeros_like(response_mask, dtype=torch.float32)
            rank = _distributed_rank()
            slot_micro_idx = int(slot.get("micro_batch_index", -1))
            for token in target_spec.get("tokens", []):
                if rank != int(token.get("owner_rank", -1)):
                    continue
                if slot_micro_idx != int(token.get("source_micro_batch_index", -2)):
                    continue
                sample_idx = int(token.get("original_sample_index", token.get("sample_index", -1)))
                position = int(token.get("position", -1))
                if (
                    0 <= sample_idx < int(response_mask.shape[0])
                    and 0 <= position < int(response_mask.shape[1])
                ):
                    mask[sample_idx, position] = 1.0
            return mask
        raise ValueError(f"unsupported_sequence_target_type:{target_type}")

    def _recompute_masked_schedule_target(
        self,
        target_spec: dict[str, Any],
        *,
        storage_dtype: str,
    ) -> tuple[dict[str, float], tuple[torch.Tensor, ...], float]:
        metrics: dict[str, float] = {}
        if not self.sequence_masked_target_enabled:
            return metrics, tuple(), 0.0
        schedule = list(self._schedule_candidates)
        if not schedule:
            metrics["global/audit/sequence_target_no_schedule_candidates"] = 1.0
            return metrics, tuple(), 0.0

        started_at = time.perf_counter()
        parameters = _trainable_parameters(self.actor)
        if not parameters:
            return metrics, tuple(), 0.0
        grad_dtypes = _parameter_grad_dtypes(parameters)
        grad_snapshot = _snapshot_parameter_grads_for_restore(parameters)
        token_mask_sum = 0.0
        contribution_scale_sum = 0.0
        effective_loss_scale_sum = 0.0
        target_type = _safe_name(target_spec.get("type", "unknown"))
        actor_config = getattr(self.actor, "config", {})
        loss_agg_mode = str(_cfg_get(actor_config, "loss_agg_mode", "token-mean"))
        apply_contribution_scale = bool(target_spec.get("apply_token_mask_contribution_scale", False))
        try:
            _finalize_fsdp_after_auxiliary_backward(self.actor)
            _clear_parameter_grads(parameters)
            micro_count = len(schedule)
            for seq_idx, slot in enumerate(schedule):
                micro_batch = slot["micro_batch"].to(get_device_id())
                response_mask = micro_batch.batch["response_mask"]
                token_mask = self._build_sequence_target_mask(
                    target_spec=target_spec,
                    slot=slot,
                    response_mask=response_mask,
                )
                contribution_scale = 1.0
                if apply_contribution_scale:
                    contribution_scale = _token_mask_contribution_scale(
                        response_mask,
                        token_mask,
                        loss_agg_mode,
                    )
                base_loss_scale = float(slot.get("loss_scale_factor", 1.0) or 1.0)
                effective_loss_scale = base_loss_scale * contribution_scale
                token_mask_sum += float(token_mask.detach().sum().item())
                contribution_scale_sum += float(contribution_scale)
                effective_loss_scale_sum += float(effective_loss_scale)
                loss = _actor_micro_batch_loss(
                    self.actor,
                    micro_batch,
                    loss_scale_factor=effective_loss_scale,
                    on_policy=bool(slot.get("on_policy", False)),
                    safe_logprob_backward=False,
                    response_mask_override=token_mask,
                )
                loss.backward()
                del loss
                try:
                    micro_batch.to("cpu")
                except Exception:
                    pass
            _finalize_fsdp_after_auxiliary_backward(self.actor)
            chunks = _snapshot_current_grad_chunks(
                self.actor,
                storage_dtype,
                grads_are_scaled=False,
            )
            local_norm_sq = _chunks_local_sumsq(chunks)
            norm_sq = _reduce_gradient_scalars(self.actor, [local_norm_sq])[0]
            (
                global_token_mask_sum,
                global_contribution_scale_sum,
                global_effective_loss_scale_sum,
            ) = _reduce_gradient_scalars(
                self.actor,
                [token_mask_sum, contribution_scale_sum, effective_loss_scale_sum],
            )
            metrics[f"global/audit/sequence_target_{target_type}_available"] = float(norm_sq > 0.0)
            metrics[f"global/audit/sequence_target_{target_type}_micro_batch_count"] = float(len(schedule))
            metrics[f"global/audit/sequence_target_{target_type}_token_mask_sum"] = float(global_token_mask_sum)
            metrics[f"global/audit/sequence_target_{target_type}_contribution_scale_sum"] = float(
                global_contribution_scale_sum
            )
            metrics[f"global/audit/sequence_target_{target_type}_effective_loss_scale_sum"] = float(
                global_effective_loss_scale_sum
            )
            metrics[f"global/full_grad_cost/sequence_target_{target_type}_seconds"] = (
                time.perf_counter() - started_at
            )
            return metrics, chunks, norm_sq
        except Exception:
            metrics[f"global/audit/sequence_target_{target_type}_error"] = 1.0
            return metrics, tuple(), 0.0
        finally:
            _finalize_fsdp_after_auxiliary_backward(self.actor)
            _restore_parameter_grads_from_snapshot(parameters, grad_snapshot, grad_dtypes=grad_dtypes)

    def _recompute_sequence_domain_targets(
        self,
    ) -> tuple[
        dict[str, float],
        dict[str, tuple[tuple[torch.Tensor, ...], float]],
        tuple[torch.Tensor, ...],
        float,
    ]:
        metrics: dict[str, float] = {}
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]] = {}
        if not self.sequence_masked_target_enabled:
            return metrics, domain_targets, tuple(), 0.0

        storage_dtype = self._domain_target_storage_dtype()
        total_metrics, total_chunks, total_norm_sq = self._recompute_masked_schedule_target(
            {"type": "total"},
            storage_dtype=storage_dtype,
        )
        metrics.update(total_metrics)
        if total_norm_sq > 0.0:
            metrics["global/audit/sequence_domain_target_available"] = 1.0
            metrics["global/full_grad_sequence/total_replay_norm"] = total_norm_sq**0.5
        for domain in self.domains:
            domain_metrics, chunks, norm_sq = self._recompute_masked_schedule_target(
                {"type": "domain", "domain": domain},
                storage_dtype=storage_dtype,
            )
            safe_domain = _safe_name(domain)
            metrics.update(
                {
                    key.replace(
                        "global/audit/sequence_target_domain",
                        f"global/audit/sequence_target_domain_{safe_domain}",
                    ).replace(
                        "global/full_grad_cost/sequence_target_domain",
                        f"global/full_grad_cost/sequence_target_domain_{safe_domain}",
                    ): value
                    for key, value in domain_metrics.items()
                }
            )
            if norm_sq > 0.0 and chunks:
                domain_targets[domain] = (chunks, norm_sq)
                metrics[f"{safe_domain}/full_grad_sequence/grad_norm"] = norm_sq**0.5

        if domain_targets and total_chunks:
            metrics.update(
                self._domain_target_closure_metrics(
                    domain_targets,
                    reference_chunks=total_chunks,
                    prefix="global/full_grad_sequence/domain_sum_vs_total",
                )
            )
        return metrics, domain_targets, total_chunks, total_norm_sq

    def _target_chunks_dot(
        self,
        left_chunks: tuple[torch.Tensor, ...],
        right_chunks: tuple[torch.Tensor, ...],
    ) -> float | None:
        if len(left_chunks) != len(right_chunks):
            return None
        local_dot = 0.0
        for left, right in zip(left_chunks, right_chunks):
            if left.numel() != right.numel():
                return None
            dot = _chunked_vector_dot(left.float(), right.float())
            if dot is not None:
                local_dot += dot
        return _reduce_gradient_scalars(self.actor, [local_dot])[0]

    def _summed_domain_target_reference_chunks(
        self,
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]],
    ) -> tuple[torch.Tensor, ...]:
        target_items = [item for item in domain_targets.values() if item[0]]
        if not target_items:
            return tuple()
        chunk_count = len(target_items[0][0])
        reference_chunks: list[torch.Tensor] = []
        for param_idx in range(chunk_count):
            summed: torch.Tensor | None = None
            for target_chunks, _norm_sq in target_items:
                if param_idx >= len(target_chunks):
                    return tuple()
                target = target_chunks[param_idx].detach().reshape(-1).to(
                    device="cpu",
                    dtype=torch.float32,
                )
                summed = target.clone() if summed is None else summed.add(target)
            if summed is not None:
                reference_chunks.append(summed)
        return tuple(reference_chunks)

    def _domain_target_closure_metrics(
        self,
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]],
        *,
        reference_chunks: tuple[torch.Tensor, ...],
        prefix: str = "global/full_grad_closure/domain_sum_vs_training",
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if not domain_targets or not reference_chunks:
            return metrics

        target_items = [item for item in domain_targets.values() if item[0]]
        if not target_items:
            return metrics

        local_ref_sq = 0.0
        local_sum_sq = 0.0
        local_diff_sq = 0.0
        local_dot = 0.0
        local_max_abs = 0.0
        for param_idx, reference in enumerate(reference_chunks):
            summed: torch.Tensor | None = None
            for target_chunks, _target_norm_sq in target_items:
                if param_idx >= len(target_chunks):
                    return metrics
                target = target_chunks[param_idx].detach().reshape(-1).to(device="cpu", dtype=torch.float32)
                summed = target.clone() if summed is None else summed.add(target)
            if summed is None:
                continue
            reference_float = reference.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
            if summed.numel() != reference_float.numel():
                return metrics
            diff = summed - reference_float
            ref_sq = _chunked_vector_dot(reference_float, reference_float)
            sum_sq = _chunked_vector_dot(summed, summed)
            diff_sq = _chunked_vector_dot(diff, diff)
            dot = _chunked_vector_dot(summed, reference_float)
            if ref_sq is not None:
                local_ref_sq += ref_sq
            if sum_sq is not None:
                local_sum_sq += sum_sq
            if diff_sq is not None:
                local_diff_sq += diff_sq
            if dot is not None:
                local_dot += dot
            if diff.numel() > 0:
                local_max_abs = max(local_max_abs, float(diff.abs().max().item()))
            del summed, reference_float, diff

        ref_sq, sum_sq, diff_sq, dot = _reduce_gradient_scalars(
            self.actor,
            [local_ref_sq, local_sum_sq, local_diff_sq, local_dot],
        )
        max_abs = _all_reduce_values_max([local_max_abs])[0]
        ref_norm = max(ref_sq, 0.0) ** 0.5
        sum_norm = max(sum_sq, 0.0) ** 0.5
        diff_norm = max(diff_sq, 0.0) ** 0.5
        metrics[f"{prefix}/rel_l2"] = diff_norm / (ref_norm + 1e-12)
        metrics[f"{prefix}/max_abs"] = max_abs
        if ref_norm > 0.0:
            metrics[f"{prefix}/norm_ratio"] = sum_norm / ref_norm
        cosine = _safe_cosine(dot, sum_norm, ref_norm)
        if cosine is not None:
            metrics[f"{prefix}/cosine"] = cosine
        if ref_sq > 0.0:
            metrics[f"{prefix}/projection_share"] = dot / ref_sq
        return metrics

    def _finish_direct_domain_gradient_metrics(
        self,
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]],
    ) -> tuple[dict[str, float], dict[str, tuple[tuple[torch.Tensor, ...], float]]]:
        metrics: dict[str, float] = {}
        available_domains = [domain for domain in self.domains if domain in domain_targets]
        if not available_domains:
            return metrics, domain_targets

        metrics["global/audit/full_gradient_domain_sequential_available"] = 1.0
        metrics["global/audit/full_gradient_domain_direct_recompute_used"] = 1.0
        for domain in available_domains:
            _chunks, norm_sq = domain_targets[domain]
            if norm_sq <= 0.0:
                continue
            safe_domain = _safe_name(domain)
            metrics[f"{safe_domain}/full_grad/grad_norm"] = norm_sq**0.5
            metrics[f"{safe_domain}/full_grad/sample_count"] = _all_reduce_sum(
                self._sample_counts.get(domain, 0)
            )

        if len(available_domains) == 1:
            metrics["global/audit/full_gradient_single_domain_target"] = 1.0
            return metrics, domain_targets

        if len(available_domains) != 2:
            return metrics, domain_targets

        first_domain, second_domain = available_domains[0], available_domains[1]
        first_chunks, first_norm_sq = domain_targets[first_domain]
        second_chunks, second_norm_sq = domain_targets[second_domain]
        if first_norm_sq <= 0.0 or second_norm_sq <= 0.0:
            return metrics, domain_targets

        first_second_dot = self._target_chunks_dot(first_chunks, second_chunks)
        if first_second_dot is None:
            return metrics, domain_targets

        first_norm = first_norm_sq**0.5
        second_norm = second_norm_sq**0.5
        total_norm_sq = max(first_norm_sq + second_norm_sq + 2.0 * first_second_dot, 0.0)
        total_norm = total_norm_sq**0.5
        first_total_dot = first_norm_sq + first_second_dot
        second_total_dot = second_norm_sq + first_second_dot
        first_safe = _safe_name(first_domain)
        second_safe = _safe_name(second_domain)
        pair = f"{first_safe}_vs_{second_safe}"

        domain_cosine = _safe_cosine(first_second_dot, first_norm, second_norm)
        if domain_cosine is not None:
            metrics[f"global/full_grad_conflict/{pair}/full_grad_cosine_train_i_k"] = domain_cosine
            metrics[f"global/full_grad_conflict/{pair}/conflict_magnitude_i_k"] = max(0.0, -domain_cosine)

        first_total_cosine = _safe_cosine(first_total_dot, first_norm, total_norm)
        if first_total_cosine is not None:
            metrics[f"global/full_grad_alignment/{first_safe}_vs_total/full_grad_cosine_domain_total"] = (
                first_total_cosine
            )
        second_total_cosine = _safe_cosine(second_total_dot, second_norm, total_norm)
        if second_total_cosine is not None:
            metrics[f"global/full_grad_alignment/{second_safe}_vs_total/full_grad_cosine_domain_total"] = (
                second_total_cosine
            )
        if total_norm_sq > 0.0:
            metrics[f"global/full_grad_contribution/{first_safe}_to_total/signed_projection_share"] = (
                first_total_dot / total_norm_sq
            )
            metrics[f"global/full_grad_contribution/{second_safe}_to_total/signed_projection_share"] = (
                second_total_dot / total_norm_sq
            )
        return metrics, domain_targets

    def _finish_domain_gradient_metrics(
        self,
        first_chunks: tuple[torch.Tensor, ...],
    ) -> tuple[dict[str, float], dict[str, tuple[tuple[torch.Tensor, ...], float]]]:
        metrics: dict[str, float] = {}
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]] = {}
        needs_domain_target_chunks = self.sample_cos_enabled or self.token_gradient_enabled
        snapshot = _current_grad_difference_snapshot(
            self.actor,
            first_chunks,
            self._domain_target_storage_dtype() if needs_domain_target_chunks else None,
        )
        if snapshot is None:
            return metrics, domain_targets
        first_norm_sq = snapshot.first_norm_sq
        total_norm_sq = snapshot.total_norm_sq
        first_total_dot = snapshot.first_total_dot
        if first_norm_sq <= 0.0 or total_norm_sq <= 0.0:
            return metrics, domain_targets

        first_domain, second_domain = self.domains[0], self.domains[1]
        first_norm = first_norm_sq**0.5
        total_norm = total_norm_sq**0.5
        second_norm_sq = max(snapshot.second_norm_sq, 0.0)
        second_norm = second_norm_sq**0.5
        first_second_dot = snapshot.first_second_dot
        second_total_dot = snapshot.second_total_dot

        first_safe = _safe_name(first_domain)
        second_safe = _safe_name(second_domain)
        metrics["global/audit/full_gradient_domain_sequential_available"] = 1.0
        metrics[f"{first_safe}/full_grad/grad_norm"] = first_norm
        metrics[f"{first_safe}/full_grad/sample_count"] = _all_reduce_sum(self._sample_counts.get(first_domain, 0))
        metrics[f"{second_safe}/full_grad/grad_norm"] = second_norm
        metrics[f"{second_safe}/full_grad/sample_count"] = _all_reduce_sum(self._sample_counts.get(second_domain, 0))
        pair = f"{first_safe}_vs_{second_safe}"
        domain_cosine = _safe_cosine(first_second_dot, first_norm, second_norm)
        if domain_cosine is not None:
            metrics[f"global/full_grad_conflict/{pair}/full_grad_cosine_train_i_k"] = domain_cosine
            metrics[f"global/full_grad_conflict/{pair}/conflict_magnitude_i_k"] = max(0.0, -domain_cosine)

        first_total_cosine = _safe_cosine(first_total_dot, first_norm, total_norm)
        if first_total_cosine is not None:
            metrics[f"global/full_grad_alignment/{first_safe}_vs_total/full_grad_cosine_domain_total"] = first_total_cosine
        second_total_cosine = _safe_cosine(second_total_dot, second_norm, total_norm)
        if second_total_cosine is not None:
            metrics[f"global/full_grad_alignment/{second_safe}_vs_total/full_grad_cosine_domain_total"] = second_total_cosine

        if total_norm_sq > 0:
            metrics[f"global/full_grad_contribution/{first_safe}_to_total/signed_projection_share"] = first_total_dot / total_norm_sq
            metrics[f"global/full_grad_contribution/{second_safe}_to_total/signed_projection_share"] = second_total_dot / total_norm_sq

        if needs_domain_target_chunks and snapshot.second_chunks is not None:
            domain_targets[first_domain] = (first_chunks, first_norm_sq)
            second_target_norm_sq = snapshot.second_target_norm_sq
            if second_target_norm_sq is not None and second_target_norm_sq > 0.0:
                domain_targets[second_domain] = (snapshot.second_chunks, second_target_norm_sq)

        return metrics, domain_targets

    def _finish_single_domain_gradient_metrics(
        self,
        domain_chunks: tuple[torch.Tensor, ...],
    ) -> tuple[dict[str, float], dict[str, tuple[tuple[torch.Tensor, ...], float]]]:
        metrics: dict[str, float] = {}
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]] = {}
        if not self.domains:
            return metrics, domain_targets
        norm_sq = 0.0
        for chunk in domain_chunks:
            chunk_float = chunk.float()
            norm_sq += _chunked_vector_dot(chunk_float, chunk_float) or 0.0
            del chunk_float
        norm_sq = _reduce_gradient_scalars(self.actor, [norm_sq])[0]
        if norm_sq <= 0.0:
            return metrics, domain_targets
        domain = self.domains[0]
        safe_domain = _safe_name(domain)
        metrics["global/audit/full_gradient_domain_sequential_available"] = 1.0
        metrics["global/audit/full_gradient_single_domain_target"] = 1.0
        metrics[f"{safe_domain}/full_grad/grad_norm"] = norm_sq**0.5
        metrics[f"{safe_domain}/full_grad/sample_count"] = _all_reduce_sum(
            self._sample_counts.get(domain, 0)
        )
        domain_targets[domain] = (domain_chunks, norm_sq)
        return metrics, domain_targets

    def _sample_norm_metrics(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        by_domain: dict[str, list[float]] = {}
        for row in self._sample_records:
            norm = row.get("sample_grad_norm")
            if norm is not None:
                by_domain.setdefault(str(row["domain"]), []).append(float(norm))
        domain_names = sorted(set(self.domains) | set(_all_gather_list(list(by_domain))))
        for domain in domain_names:
            finite = _finite_values(_all_gather_list(by_domain.get(domain, [])))
            if not finite:
                continue
            safe_domain = _safe_name(domain)
            std = _std(finite) or 0.0
            mean = _mean(finite) or 0.0
            metrics[f"{safe_domain}/sample_grad/norm_mean"] = mean
            metrics[f"{safe_domain}/sample_grad/norm_p50"] = _percentile(finite, 50.0) or 0.0
            metrics[f"{safe_domain}/sample_grad/norm_p95"] = _percentile(finite, 95.0) or 0.0
            metrics[f"{safe_domain}/sample_grad/norm_max"] = max(finite)
            metrics[f"{safe_domain}/sample_grad/norm_cv"] = std / (abs(mean) + 1e-12)
            metrics[f"{safe_domain}/sample_grad/sample_count"] = float(len(finite))
        return metrics

    def _sample_cos_metrics(self, domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]]) -> dict[str, float]:
        metrics: dict[str, float] = {}
        parameters_for_final_restore: tuple[torch.nn.Parameter, ...] = tuple()
        final_grad_snapshot: tuple[torch.Tensor | None, ...] | None = None
        final_grad_dtypes: tuple[torch.dtype | None, ...] = tuple()
        if self.sample_gradient_backward_recompute_enabled:
            try:
                parameters_for_final_restore = _trainable_parameters(self.actor)
                final_grad_dtypes = _parameter_grad_dtypes(parameters_for_final_restore)
                final_grad_snapshot = _snapshot_parameter_grads_for_restore(parameters_for_final_restore)
            except Exception:
                parameters_for_final_restore = tuple()
                final_grad_snapshot = None
                final_grad_dtypes = tuple()
        structural_unavailable_reason: str | None = None
        try:
            for domain in sorted(domain_targets):
                candidates = self._sample_candidates.get(domain, [])
                sync_sample_backward = (
                    self.sample_gradient_backward_recompute_enabled
                    and self.sample_gradient_backward_sync_enabled
                )
                if not candidates and not sync_sample_backward:
                    continue
                target_chunks, target_norm_sq = domain_targets[domain]
                target_norm = target_norm_sq**0.5
                cos_values: list[float] = []
                share_values: list[float] = []
                raw_share_values: list[float] = []
                share_scale_values: list[float] = []
                local_share_rows: list[tuple[dict[str, Any], float]] = []
                availability_values: list[float] = []
                disconnected_values: list[float] = []
                seconds_values: list[float] = []
                backward_used_values: list[float] = []
                backward_sync_values: list[float] = []
                restore_pre_rel_l2_values: list[float] = []
                restore_post_rel_l2_values: list[float] = []
                if sync_sample_backward:
                    candidate_stats, sync_metrics = self._recompute_sync_sample_domain_stats(
                        domain,
                        candidates,
                        target_chunks=target_chunks,
                        target_norm=target_norm,
                        target_norm_sq=target_norm_sq,
                    )
                    metrics.update(sync_metrics)
                else:
                    candidate_stats = []
                    for candidate in candidates:
                        row = candidate["row"]
                        if structural_unavailable_reason is None:
                            stats = self._recompute_sample_to_domain_stats(
                                candidate["micro_batch"],
                                target_chunks=target_chunks,
                                target_norm=target_norm,
                                target_norm_sq=target_norm_sq,
                                loss_scale_factor=float(row.get("loss_scale_factor", 1.0)),
                                on_policy=bool(row.get("on_policy", False)),
                            )
                            if stats.get("sample_recompute_autograd_error") == "all_parameters_disconnected":
                                structural_unavailable_reason = "all_parameters_disconnected"
                        else:
                            stats = {
                                "sample_to_domain_cos": None,
                                "sample_projection_share": None,
                                "sample_projection_share_normalized": None,
                                "sample_recompute_grad_norm": 0.0,
                                "sample_recompute_non_none_grad_count": 0.0,
                                "sample_recompute_available": 0.0,
                                "sample_recompute_autograd_error": structural_unavailable_reason,
                                "sample_recompute_backward_used": float(
                                    self.sample_gradient_backward_recompute_enabled
                                ),
                                "sample_recompute_backward_sync_used": float(
                                    self.sample_gradient_backward_sync_enabled
                                ),
                            }
                        candidate_stats.append((row, stats))
                for row, stats in candidate_stats:
                    row["computed_for_cos"] = True
                    row.update(stats)
                    if row.get("sample_grad_norm") is None and stats.get("sample_recompute_grad_norm") is not None:
                        row["sample_grad_norm"] = stats.get("sample_recompute_grad_norm")
                    cos_value = stats.get("sample_to_domain_cos")
                    share_value = stats.get("sample_projection_share")
                    disconnected_values.append(
                        float(stats.get("sample_recompute_autograd_error") == "all_parameters_disconnected")
                    )
                    available = float(
                        stats.get(
                            "sample_recompute_available",
                            cos_value is not None or share_value is not None,
                        )
                    )
                    availability_values.append(available)
                    seconds_values.extend(_finite_values([stats.get("sample_recompute_seconds")]))
                    backward_used_values.extend(_finite_values([stats.get("sample_recompute_backward_used")]))
                    backward_sync_values.extend(_finite_values([stats.get("sample_recompute_backward_sync_used")]))
                    restore_pre_rel_l2_values.extend(
                        _finite_values([stats.get("sample_recompute_restore_pre_target_rel_l2")])
                    )
                    restore_post_rel_l2_values.extend(
                        _finite_values([stats.get("sample_recompute_restore_post_target_rel_l2")])
                    )
                    if cos_value is not None:
                        cos_values.append(float(cos_value))
                    if share_value is not None:
                        share_float = float(share_value)
                        share_values.append(share_float)
                        local_share_rows.append((row, share_float))
                        raw_share_value = stats.get("sample_projection_share_raw", share_value)
                        raw_share_values.extend(_finite_values([raw_share_value]))
                    share_scale_values.extend(_finite_values([stats.get("sample_projection_share_scale")]))
                safe_domain = _safe_name(domain)
                global_cos_values = _finite_values(_all_gather_list(cos_values))
                global_share_values = _finite_values(_all_gather_list(share_values))
                global_raw_share_values = _finite_values(_all_gather_list(raw_share_values))
                global_share_scale_values = _finite_values(_all_gather_list(share_scale_values))
                global_availability_values = _finite_values(_all_gather_list(availability_values))
                global_disconnected_values = _finite_values(_all_gather_list(disconnected_values))
                global_seconds_values = _finite_values(_all_gather_list(seconds_values))
                global_backward_used_values = _finite_values(_all_gather_list(backward_used_values))
                global_backward_sync_values = _finite_values(_all_gather_list(backward_sync_values))
                global_restore_pre_rel_l2_values = _finite_values(_all_gather_list(restore_pre_rel_l2_values))
                global_restore_post_rel_l2_values = _finite_values(_all_gather_list(restore_post_rel_l2_values))
                available_count = 0
                attempted_count = 0
                if global_availability_values:
                    attempted_count = len(global_availability_values)
                    available_count = sum(value > 0.5 for value in global_availability_values)
                    metrics[f"{safe_domain}/sample_grad_cos/attempted_count"] = float(attempted_count)
                    metrics[f"{safe_domain}/sample_grad_cos/unavailable_count"] = float(
                        attempted_count - available_count
                    )
                    metrics[f"{safe_domain}/sample_grad_cos/valid_frac"] = available_count / attempted_count
                if global_disconnected_values:
                    metrics[f"{safe_domain}/sample_grad_cos/all_parameters_disconnected_count"] = float(
                        sum(value > 0.5 for value in global_disconnected_values)
                    )
                if global_seconds_values:
                    metrics[f"{safe_domain}/sample_grad_cost/seconds_sum"] = sum(global_seconds_values)
                    metrics[f"{safe_domain}/sample_grad_cost/seconds_mean"] = _mean(global_seconds_values) or 0.0
                if global_backward_used_values:
                    metrics[f"{safe_domain}/sample_grad_cost/backward_recompute_count"] = float(
                        sum(value > 0.5 for value in global_backward_used_values)
                    )
                if global_backward_sync_values:
                    metrics[f"{safe_domain}/sample_grad_cost/backward_sync_count"] = float(
                        sum(value > 0.5 for value in global_backward_sync_values)
                    )
                if global_restore_pre_rel_l2_values:
                    metrics[f"{safe_domain}/sample_grad_cost/restore_pre_target_rel_l2_max"] = max(
                        global_restore_pre_rel_l2_values
                    )
                if global_restore_post_rel_l2_values:
                    metrics[f"{safe_domain}/sample_grad_cost/restore_post_target_rel_l2_max"] = max(
                        global_restore_post_rel_l2_values
                    )
                if global_cos_values:
                    metrics[f"{safe_domain}/sample_grad_cos/domain_cos_mean"] = _mean(global_cos_values) or 0.0
                    metrics[f"{safe_domain}/sample_grad_cos/domain_cos_p05"] = _percentile(global_cos_values, 5.0) or 0.0
                    metrics[f"{safe_domain}/sample_grad_cos/domain_cos_negative_frac"] = float(
                        np.mean([value < 0.0 for value in global_cos_values])
                    )
                    metrics[f"{safe_domain}/sample_grad_cos/sample_count"] = float(len(global_cos_values))
                if global_share_values:
                    metrics[f"{safe_domain}/sample_grad_contribution/projection_share_mean"] = _mean(global_share_values) or 0.0
                    metrics[f"{safe_domain}/sample_grad_contribution/projection_share_min"] = min(global_share_values)
                    metrics[f"{safe_domain}/sample_grad_contribution/projection_share_max"] = max(global_share_values)
                    metrics[f"{safe_domain}/sample_grad_contribution/projection_share_negative_frac"] = float(
                        np.mean([value < 0.0 for value in global_share_values])
                    )
                    metrics[f"{safe_domain}/sample_grad_contribution/top1_abs_share"] = max(
                        abs(value) for value in global_share_values
                    )
                    replica_count = (
                        _gradient_replica_count(self.actor)
                        if self._sample_gradient_uses_full_local_params
                        else 1
                    )
                    raw_values_for_sum = global_raw_share_values or global_share_values
                    projection_share_sum_across_replicas = sum(raw_values_for_sum)
                    projection_share_sum = sum(global_share_values)
                    normalized_share_values: list[float] = []
                    if abs(projection_share_sum) > 1e-12:
                        normalized_share_values = [
                            value / projection_share_sum for value in global_share_values
                        ]
                        for row, share_value in local_share_rows:
                            row["sample_projection_share_normalized"] = share_value / projection_share_sum
                    else:
                        for row, _share_value in local_share_rows:
                            row["sample_projection_share_normalized"] = None
                    raw_expected_sum = 1.0
                    if global_share_scale_values:
                        scale_mean = _mean(global_share_scale_values) or 0.0
                        if scale_mean > 0.0:
                            raw_expected_sum = 1.0 / scale_mean
                    metrics[
                        f"{safe_domain}/sample_grad_contribution/projection_share_sum_across_replicas"
                    ] = projection_share_sum_across_replicas
                    metrics[f"{safe_domain}/sample_grad_contribution/projection_share_sum_raw"] = (
                        projection_share_sum_across_replicas
                    )
                    metrics[f"{safe_domain}/sample_grad_contribution/projection_share_sum_raw_expected"] = (
                        raw_expected_sum
                    )
                    metrics[f"{safe_domain}/sample_grad_contribution/projection_share_sum_raw_error"] = abs(
                        projection_share_sum_across_replicas - raw_expected_sum
                    )
                    metrics[
                        f"{safe_domain}/sample_grad_contribution/projection_share_replica_count"
                    ] = float(replica_count)
                    if global_share_scale_values:
                        metrics[f"{safe_domain}/sample_grad_contribution/projection_share_scale_mean"] = (
                            _mean(global_share_scale_values) or 0.0
                        )
                    metrics[f"{safe_domain}/sample_grad_contribution/projection_share_sum"] = projection_share_sum
                    metrics[f"{safe_domain}/sample_grad_contribution/projection_share_sum_error"] = abs(
                        projection_share_sum - 1.0
                    )
                    if normalized_share_values:
                        normalized_share_sum = sum(normalized_share_values)
                        metrics[
                            f"{safe_domain}/sample_grad_contribution/projection_share_normalized_mean"
                        ] = _mean(normalized_share_values) or 0.0
                        metrics[
                            f"{safe_domain}/sample_grad_contribution/projection_share_normalized_min"
                        ] = min(normalized_share_values)
                        metrics[
                            f"{safe_domain}/sample_grad_contribution/projection_share_normalized_max"
                        ] = max(normalized_share_values)
                        metrics[
                            f"{safe_domain}/sample_grad_contribution/projection_share_normalized_negative_frac"
                        ] = float(np.mean([value < 0.0 for value in normalized_share_values]))
                        metrics[
                            f"{safe_domain}/sample_grad_contribution/top1_abs_share_normalized"
                        ] = max(abs(value) for value in normalized_share_values)
                        metrics[
                            f"{safe_domain}/sample_grad_contribution/projection_share_normalized_sum"
                        ] = normalized_share_sum
                        metrics[
                            f"{safe_domain}/sample_grad_contribution/projection_share_normalized_sum_error"
                        ] = abs(normalized_share_sum - 1.0)
                    metrics[f"{safe_domain}/sample_grad_closure/projection_share_sum"] = projection_share_sum
                    metrics[f"{safe_domain}/sample_grad_closure/projection_share_sum_error"] = abs(
                        projection_share_sum - 1.0
                    )
                    if normalized_share_values:
                        normalized_share_sum = sum(normalized_share_values)
                        metrics[
                            f"{safe_domain}/sample_grad_closure/projection_share_normalized_sum"
                        ] = normalized_share_sum
                        metrics[
                            f"{safe_domain}/sample_grad_closure/projection_share_normalized_sum_error"
                        ] = abs(normalized_share_sum - 1.0)
                    metrics[f"{safe_domain}/sample_grad_closure/projection_share_sum_raw"] = (
                        projection_share_sum_across_replicas
                    )
                    metrics[f"{safe_domain}/sample_grad_closure/projection_share_sum_raw_expected"] = raw_expected_sum
                    metrics[f"{safe_domain}/sample_grad_closure/projection_share_sum_raw_error"] = abs(
                        projection_share_sum_across_replicas - raw_expected_sum
                    )
                if attempted_count > 0:
                    metrics[f"{safe_domain}/sample_grad_closure/valid_frac"] = available_count / attempted_count
        finally:
            if final_grad_snapshot is not None and parameters_for_final_restore:
                _restore_parameter_grads_from_snapshot(
                    parameters_for_final_restore,
                    final_grad_snapshot,
                    grad_dtypes=final_grad_dtypes,
                )
        return metrics

    def _sample_sync_template_micro_batch(self) -> DataProto | None:
        for candidates in self._sample_candidates.values():
            for candidate in candidates:
                micro_batch = candidate.get("micro_batch")
                if micro_batch is not None:
                    return micro_batch
        for candidates in self._domain_recompute_candidates.values():
            for candidate in candidates:
                micro_batch = candidate.get("micro_batch")
                if micro_batch is not None:
                    return micro_batch
        for candidates in self._token_gradient_candidates.values():
            for candidate in candidates:
                micro_batch = candidate.get("micro_batch")
                if micro_batch is not None:
                    return micro_batch
        return None

    def _unavailable_sample_recompute_stats(self, reason: str) -> dict[str, float | str | None]:
        return {
            "sample_to_domain_cos": None,
            "sample_projection_share": None,
            "sample_projection_share_normalized": None,
            "sample_projection_share_raw": None,
            "sample_projection_share_scale": 1.0,
            "sample_recompute_grad_norm": None,
            "sample_recompute_grad_norm_raw": None,
            "sample_recompute_non_none_grad_count": 0.0,
            "sample_recompute_available": 0.0,
            "sample_recompute_autograd_error": reason,
            "sample_recompute_backward_used": float(self.sample_gradient_backward_recompute_enabled),
            "sample_recompute_backward_sync_used": float(self.sample_gradient_backward_sync_enabled),
            "sample_recompute_replica_count": float(_gradient_replica_count(self.actor)),
            "sample_recompute_seconds": 0.0,
            "sample_recompute_restore_pre_target_rel_l2": 0.0,
            "sample_recompute_restore_pre_target_max_abs": 0.0,
            "sample_recompute_restore_post_target_rel_l2": 0.0,
            "sample_recompute_restore_post_target_max_abs": 0.0,
            "sample_recompute_restore_target_norm": 0.0,
        }

    def _sequence_sample_slot_stats(
        self,
        *,
        slot: dict[str, Any],
        target_chunks: tuple[torch.Tensor, ...],
        target_norm: float,
        target_norm_sq: float,
    ) -> tuple[dict[str, Any], tuple[torch.Tensor, ...]]:
        started_at = time.perf_counter()
        target_spec = {
            "type": "sample",
            "owner_rank": int(slot.get("owner_rank", -1)),
            "source_micro_batch_index": int(slot.get("source_micro_batch_index", -1)),
            "sample_index": int(slot.get("sample_index", 0)),
            "apply_token_mask_contribution_scale": True,
        }
        _metrics, chunks, norm_sq = self._recompute_masked_schedule_target(
            target_spec,
            storage_dtype="float32",
        )
        token_mask_sum = float(
            _metrics.get("global/audit/sequence_target_sample_token_mask_sum", 0.0) or 0.0
        )
        effective_loss_scale_sum = float(
            _metrics.get("global/audit/sequence_target_sample_effective_loss_scale_sum", 0.0) or 0.0
        )
        autograd_error: str | None = None
        dot = None
        is_owner = _distributed_rank() == int(slot.get("owner_rank", -1))
        if chunks and norm_sq > 0.0:
            dot = self._target_chunks_dot(chunks, target_chunks)
        if not chunks or norm_sq <= 0.0 or dot is None:
            autograd_error = "sequence_sample_target_unavailable"
            return {
                "sample_to_domain_cos": None,
                "sample_projection_share": None,
                "sample_projection_share_normalized": None,
                "sample_projection_share_raw": None,
                "sample_projection_share_scale": 1.0,
                "sample_recompute_grad_norm": None,
                "sample_recompute_grad_norm_raw": None,
                "sample_recompute_local_grad_norm": None,
                "sample_recompute_non_none_grad_count": 0.0,
                "sample_recompute_available": 0.0,
                "sample_recompute_autograd_error": autograd_error,
                "sample_recompute_backward_used": 1.0,
                "sample_recompute_backward_sync_used": 1.0,
                "sample_recompute_gradients_synced_by_backward": 1.0,
                "sample_recompute_replica_count": float(_gradient_replica_count(self.actor)),
                "sample_recompute_seconds": time.perf_counter() - started_at,
                "sample_recompute_token_mask_sum": token_mask_sum,
                "sample_recompute_effective_loss_scale_factor": effective_loss_scale_sum,
                "sample_recompute_is_owner": float(is_owner),
                "sample_recompute_sequence_replay_used": 1.0,
                "sample_recompute_restore_pre_target_rel_l2": 0.0,
                "sample_recompute_restore_pre_target_max_abs": 0.0,
                "sample_recompute_restore_post_target_rel_l2": 0.0,
                "sample_recompute_restore_post_target_max_abs": 0.0,
                "sample_recompute_restore_target_norm": target_norm,
            }, tuple()

        sample_norm = max(norm_sq, 0.0) ** 0.5
        available = sample_norm > 0.0 and target_norm > 0.0
        projection_share = dot / target_norm_sq if available and target_norm_sq > 0.0 else None
        cosine = dot / (sample_norm * target_norm) if available else None
        return {
            "sample_to_domain_cos": cosine,
            "sample_projection_share": projection_share,
            "sample_projection_share_normalized": None,
            "sample_projection_share_raw": projection_share,
            "sample_projection_share_scale": 1.0,
            "sample_recompute_grad_norm": sample_norm,
            "sample_recompute_grad_norm_raw": sample_norm,
            "sample_recompute_local_grad_norm": sample_norm,
            "sample_recompute_non_none_grad_count": float(len(_trainable_parameters(self.actor))),
            "sample_recompute_available": float(available),
            "sample_recompute_autograd_error": None,
            "sample_recompute_backward_used": 1.0,
            "sample_recompute_backward_sync_used": 1.0,
            "sample_recompute_gradients_synced_by_backward": 1.0,
            "sample_recompute_replica_count": float(_gradient_replica_count(self.actor)),
            "sample_recompute_seconds": time.perf_counter() - started_at,
            "sample_recompute_token_mask_sum": token_mask_sum,
            "sample_recompute_effective_loss_scale_factor": effective_loss_scale_sum,
            "sample_recompute_is_owner": float(is_owner),
            "sample_recompute_sequence_replay_used": 1.0,
            "sample_recompute_restore_pre_target_rel_l2": 0.0,
            "sample_recompute_restore_pre_target_max_abs": 0.0,
            "sample_recompute_restore_post_target_rel_l2": 0.0,
            "sample_recompute_restore_post_target_max_abs": 0.0,
            "sample_recompute_restore_target_norm": target_norm,
        }, chunks

    def _recompute_sequence_sample_domain_stats(
        self,
        domain: str,
        candidates: list[dict[str, Any]],
        *,
        target_chunks: tuple[torch.Tensor, ...],
        target_norm: float,
        target_norm_sq: float,
    ) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, float]]:
        metrics: dict[str, float] = {}
        safe_domain = _safe_name(domain)
        local_rank = _distributed_rank()
        local_slots = [
            {
                "domain": domain,
                "owner_rank": local_rank,
                "local_index": idx,
                "sample_id": str(candidate["row"].get("sample_id", f"rank{local_rank}:sample{idx}")),
                "source_micro_batch_index": int(candidate["row"].get("micro_batch_index", idx)),
                "sample_index": int(candidate["row"].get("sample_index", 0) or 0),
                "loss_scale_factor": float(candidate["row"].get("loss_scale_factor", 1.0)),
                "on_policy": bool(candidate["row"].get("on_policy", False)),
            }
            for idx, candidate in enumerate(candidates)
        ]
        global_slots = sorted(
            _all_gather_list(local_slots),
            key=lambda item: (
                str(item.get("domain", "")),
                int(item.get("owner_rank", 0)),
                int(item.get("source_micro_batch_index", 0)),
                int(item.get("sample_index", 0)),
                str(item.get("sample_id", "")),
            ),
        )
        if not global_slots:
            return [], metrics

        local_by_index = {idx: candidate for idx, candidate in enumerate(candidates)}
        parameters = _trainable_parameters(self.actor)
        sample_sum_chunks: list[torch.Tensor] | None = None
        row_stats: list[tuple[dict[str, Any], dict[str, Any]]] = []
        slot_error_count = 0
        sequence_seconds: list[float] = []

        for slot in global_slots:
            stats, chunks = self._sequence_sample_slot_stats(
                slot=slot,
                target_chunks=target_chunks,
                target_norm=target_norm,
                target_norm_sq=target_norm_sq,
            )
            if chunks:
                if sample_sum_chunks is None:
                    sample_sum_chunks = [chunk.detach().reshape(-1).float().clone() for chunk in chunks]
                elif len(sample_sum_chunks) == len(chunks):
                    for idx, chunk in enumerate(chunks):
                        sample_sum_chunks[idx].add_(chunk.detach().reshape(-1).float())
            if stats.get("sample_recompute_autograd_error"):
                slot_error_count += 1
            sequence_seconds.extend(_finite_values([stats.get("sample_recompute_seconds")]))
            if int(slot.get("owner_rank", -1)) == local_rank:
                local_candidate = local_by_index.get(int(slot.get("local_index", -1)))
                if local_candidate is not None:
                    row_stats.append((local_candidate["row"], stats))

        global_sequence_seconds = _finite_values(_all_gather_list(sequence_seconds))
        metrics[f"{safe_domain}/sample_sequence_closure/vector_slot_count"] = float(len(global_slots))
        metrics[f"{safe_domain}/sample_sequence_closure/vector_error_count"] = float(slot_error_count)
        if global_sequence_seconds:
            metrics[f"{safe_domain}/sample_grad_cost/sequence_seconds_sum"] = sum(global_sequence_seconds)
            metrics[f"{safe_domain}/sample_grad_cost/sequence_seconds_mean"] = (
                _mean(global_sequence_seconds) or 0.0
            )

        if sample_sum_chunks is not None:
            closure_stats, _ = _gradient_chunk_pair_stats(
                self.actor,
                target_chunks,
                tuple(sample_sum_chunks),
                parameters,
            )
            for key, value in closure_stats.items():
                metrics[f"{safe_domain}/sample_sequence_closure/sum_vs_domain_{key}"] = float(value)
                metrics[f"{safe_domain}/sample_grad_closure/vector_{key}"] = float(value)
            rel_l2 = closure_stats.get("rel_l2")
            cosine = closure_stats.get("cosine")
            trusted = (
                rel_l2 is not None
                and float(rel_l2) <= self.domain_direct_recompute_closure_rel_l2_threshold
                and cosine is not None
                and float(cosine) > 0.999
            )
            metrics[f"{safe_domain}/sample_sequence_closure/vector_available"] = 1.0
            metrics[f"{safe_domain}/sample_grad_closure/vector_available"] = 1.0
            metrics[f"{safe_domain}/sample_grad_contribution/projection_share_trusted"] = float(trusted)
        else:
            metrics[f"{safe_domain}/sample_sequence_closure/vector_available"] = 0.0
            metrics[f"{safe_domain}/sample_grad_closure/vector_available"] = 0.0
            metrics[f"{safe_domain}/sample_grad_contribution/projection_share_trusted"] = 0.0
        return row_stats, metrics

    def _recompute_sync_sample_domain_stats(
        self,
        domain: str,
        candidates: list[dict[str, Any]],
        *,
        target_chunks: tuple[torch.Tensor, ...],
        target_norm: float,
        target_norm_sq: float,
    ) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], dict[str, float]]:
        if self.sequence_masked_target_enabled and self.sequence_masked_target_use_as_primary:
            return self._recompute_sequence_sample_domain_stats(
                domain,
                candidates,
                target_chunks=target_chunks,
                target_norm=target_norm,
                target_norm_sq=target_norm_sq,
            )
        metrics: dict[str, float] = {}
        safe_domain = _safe_name(domain)
        local_rank = _distributed_rank()
        local_by_index = {idx: candidate for idx, candidate in enumerate(candidates)}
        local_slots = [
            {
                "domain": domain,
                "owner_rank": local_rank,
                "local_index": idx,
                "sample_id": str(candidate["row"].get("sample_id", f"rank{local_rank}:sample{idx}")),
                "loss_scale_factor": float(candidate["row"].get("loss_scale_factor", 1.0)),
                "on_policy": bool(candidate["row"].get("on_policy", False)),
            }
            for idx, candidate in enumerate(candidates)
        ]
        global_slots = sorted(
            _all_gather_list(local_slots),
            key=lambda item: (
                str(item.get("domain", "")),
                int(item.get("owner_rank", 0)),
                int(item.get("local_index", 0)),
                str(item.get("sample_id", "")),
            ),
        )
        if not global_slots:
            return [], metrics

        template_micro_batch = self._sample_sync_template_micro_batch()
        if not _all_ranks_true(template_micro_batch is not None):
            metrics[f"{safe_domain}/sample_grad_closure/vector_available"] = 0.0
            metrics[f"{safe_domain}/sample_grad_closure/vector_slot_count"] = float(len(global_slots))
            return [
                (candidate["row"], self._unavailable_sample_recompute_stats("sync_template_unavailable"))
                for candidate in candidates
            ], metrics

        parameters = _trainable_parameters(self.actor)
        sample_sum_chunks: list[torch.Tensor] | None = None
        row_stats: list[tuple[dict[str, Any], dict[str, Any]]] = []
        slot_error_count = 0
        for slot in global_slots:
            owner_rank = int(slot.get("owner_rank", -1))
            local_index = int(slot.get("local_index", -1))
            local_candidate = local_by_index.get(local_index) if owner_rank == local_rank else None
            stats, chunks = self._recompute_sync_sample_slot_to_domain_stats(
                slot,
                local_candidate,
                template_micro_batch=template_micro_batch,
                target_chunks=target_chunks,
                target_norm=target_norm,
                target_norm_sq=target_norm_sq,
            )
            if chunks:
                if sample_sum_chunks is None:
                    sample_sum_chunks = [chunk.detach().reshape(-1).float().clone() for chunk in chunks]
                elif len(sample_sum_chunks) == len(chunks):
                    for idx, chunk in enumerate(chunks):
                        sample_sum_chunks[idx].add_(chunk.detach().reshape(-1).float())
            if stats.get("sample_recompute_autograd_error"):
                slot_error_count += 1
            if local_candidate is not None:
                row_stats.append((local_candidate["row"], stats))

        metrics[f"{safe_domain}/sample_grad_closure/vector_slot_count"] = float(len(global_slots))
        metrics[f"{safe_domain}/sample_grad_closure/vector_error_count"] = float(slot_error_count)
        if sample_sum_chunks is not None:
            closure_stats, _ = _gradient_chunk_pair_stats(
                self.actor,
                target_chunks,
                tuple(sample_sum_chunks),
                parameters,
            )
            for key, value in closure_stats.items():
                metrics[f"{safe_domain}/sample_grad_closure/vector_{key}"] = float(value)
            rel_l2 = closure_stats.get("rel_l2")
            cosine = closure_stats.get("cosine")
            trusted = (
                rel_l2 is not None
                and float(rel_l2) <= self.domain_direct_recompute_closure_rel_l2_threshold
                and cosine is not None
                and float(cosine) > 0.999
            )
            metrics[f"{safe_domain}/sample_grad_closure/vector_available"] = 1.0
            metrics[f"{safe_domain}/sample_grad_contribution/projection_share_trusted"] = float(trusted)
        else:
            metrics[f"{safe_domain}/sample_grad_closure/vector_available"] = 0.0
            metrics[f"{safe_domain}/sample_grad_contribution/projection_share_trusted"] = 0.0
        return row_stats, metrics

    def _recompute_sync_sample_slot_to_domain_stats(
        self,
        slot: dict[str, Any],
        local_candidate: dict[str, Any] | None,
        *,
        template_micro_batch: DataProto | None,
        target_chunks: tuple[torch.Tensor, ...],
        target_norm: float,
        target_norm_sq: float,
    ) -> tuple[dict[str, Any], tuple[torch.Tensor, ...]]:
        started_at = time.perf_counter()
        parameters = _trainable_parameters(self.actor)
        if template_micro_batch is None or len(parameters) != len(target_chunks):
            return self._unavailable_sample_recompute_stats("parameter_target_mismatch"), tuple()

        actor_config = getattr(self.actor, "config", {})
        loss_agg_mode = str(_cfg_get(actor_config, "loss_agg_mode", "token-mean"))
        is_owner = local_candidate is not None
        micro_batch = local_candidate["micro_batch"] if is_owner else template_micro_batch
        row = local_candidate["row"] if is_owner else {}
        loss_scale_factor = (
            float(row.get("loss_scale_factor", 1.0))
            if is_owner
            else float(slot.get("loss_scale_factor", 1.0) or 1.0)
        )
        on_policy = bool(row.get("on_policy", slot.get("on_policy", False)))
        autograd_error: str | None = None
        non_none_grad_count = 0
        local_norm_sq: float | None = None
        local_dot: float | None = None
        chunks: tuple[torch.Tensor, ...] = tuple()
        replica_count = _gradient_replica_count(self.actor)
        gradients_synced_by_backward = bool(
            self.sample_gradient_backward_recompute_enabled
            and self.sample_gradient_backward_sync_enabled
            and replica_count > 1
        )
        token_mask_sum = 0.0
        effective_loss_scale_factor = 1.0
        try:
            response_mask = micro_batch.batch["response_mask"]
            if is_owner:
                token_mask = response_mask.detach().float().clone()
                contribution_scale = _token_mask_contribution_scale(
                    response_mask,
                    token_mask,
                    loss_agg_mode,
                )
                if contribution_scale <= 0.0:
                    token_mask = torch.zeros_like(response_mask, dtype=torch.float32)
                    effective_loss_scale_factor = 1.0
                    autograd_error = "zero_contribution_scale"
                else:
                    effective_loss_scale_factor = loss_scale_factor * contribution_scale
            else:
                token_mask = torch.zeros_like(response_mask, dtype=torch.float32)
                effective_loss_scale_factor = 1.0
            token_mask_sum = float(token_mask.detach().sum().item())

            _clear_parameter_grads(parameters)
            loss = _actor_micro_batch_loss(
                self.actor,
                micro_batch,
                loss_scale_factor=effective_loss_scale_factor,
                on_policy=on_policy,
                safe_logprob_backward=False,
                response_mask_override=token_mask,
            )
            loss.backward()
            del loss
            _finalize_fsdp_after_auxiliary_backward(self.actor)
            gradients = tuple(parameter.grad for parameter in parameters)
            non_none_grad_count = sum(gradient is not None for gradient in gradients)
            local_norm_sq, local_dot = self._grad_stats_from_tensors(gradients, target_chunks)
            chunks = _snapshot_current_grad_chunks(
                self.actor,
                "float32",
                grads_are_scaled=False,
            )
        except Exception as exc:
            autograd_error = autograd_error or type(exc).__name__
        finally:
            _finalize_fsdp_after_auxiliary_backward(self.actor)
            try:
                micro_batch.to("cpu")
            except Exception:
                pass

        if local_norm_sq is None or local_dot is None:
            stats = self._unavailable_sample_recompute_stats(
                autograd_error or "parameter_target_mismatch"
            )
            stats["sample_recompute_seconds"] = time.perf_counter() - started_at
            stats["sample_recompute_non_none_grad_count"] = float(non_none_grad_count)
            stats["sample_recompute_local_grad_norm"] = (
                max(float(local_norm_sq), 0.0) ** 0.5 if local_norm_sq is not None else 0.0
            )
            stats["sample_recompute_gradients_synced_by_backward"] = float(gradients_synced_by_backward)
            stats["sample_recompute_replica_count"] = float(replica_count)
            stats["sample_recompute_is_owner"] = float(is_owner)
            stats["sample_recompute_token_mask_sum"] = token_mask_sum
            stats["sample_recompute_effective_loss_scale_factor"] = effective_loss_scale_factor
            return stats, chunks

        reduced = _all_reduce_values_sum(
            [
                float(max(local_norm_sq, 0.0)),
                float(local_dot),
                float(non_none_grad_count),
                float(len(parameters)),
            ]
        )
        norm_sq = max(reduced[0], 0.0)
        dot = reduced[1]
        count_scale = 1.0 / float(replica_count) if replica_count > 1 else 1.0
        global_non_none_grad_count = reduced[2] * count_scale
        local_grad_norm = max(float(local_norm_sq), 0.0) ** 0.5
        sample_norm = norm_sq**0.5
        available = (
            is_owner
            and autograd_error is None
            and global_non_none_grad_count > 0
            and sample_norm > 0.0
            and target_norm > 0.0
        )
        cosine = dot / (sample_norm * target_norm) if available else None
        projection_share = dot / target_norm_sq if available and target_norm_sq > 0.0 else None
        return {
            "sample_to_domain_cos": cosine,
            "sample_projection_share": projection_share,
            "sample_projection_share_normalized": None,
            "sample_projection_share_raw": projection_share,
            "sample_projection_share_scale": 1.0,
            "sample_recompute_grad_norm": sample_norm if available else None,
            "sample_recompute_grad_norm_raw": sample_norm if available else None,
            "sample_recompute_local_grad_norm": local_grad_norm,
            "sample_recompute_non_none_grad_count": float(global_non_none_grad_count),
            "sample_recompute_available": float(available),
            "sample_recompute_autograd_error": autograd_error,
            "sample_recompute_backward_used": float(self.sample_gradient_backward_recompute_enabled),
            "sample_recompute_backward_sync_used": 1.0,
            "sample_recompute_gradients_synced_by_backward": float(gradients_synced_by_backward),
            "sample_recompute_replica_count": float(replica_count),
            "sample_recompute_is_owner": float(is_owner),
            "sample_recompute_token_mask_sum": token_mask_sum,
            "sample_recompute_effective_loss_scale_factor": effective_loss_scale_factor,
            "sample_recompute_seconds": time.perf_counter() - started_at,
            "sample_recompute_restore_pre_target_rel_l2": 0.0,
            "sample_recompute_restore_pre_target_max_abs": 0.0,
            "sample_recompute_restore_post_target_rel_l2": 0.0,
            "sample_recompute_restore_post_target_max_abs": 0.0,
            "sample_recompute_restore_target_norm": target_norm,
        }, chunks

    def _token_gradient_metrics(
        self,
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]],
    ) -> dict[str, float]:
        started_at = time.perf_counter()
        metrics: dict[str, float] = {}
        rows: list[dict[str, Any]] = []
        local_rank = _distributed_rank()
        world_size = self._distributed_world_size
        parameters_for_final_restore = _trainable_parameters(self.actor)
        final_grad_dtypes = _parameter_grad_dtypes(parameters_for_final_restore)
        final_grad_snapshot = _snapshot_parameter_grads_for_restore(parameters_for_final_restore)
        token_recompute_attempted = False
        local_records_by_key: dict[tuple[int, int], dict[str, Any]] = {}
        local_metadata: list[dict[str, Any]] = []
        local_supported_targets = [str(domain) for domain in domain_targets]
        supported_target_counts: dict[str, int] = {}
        for domain in _all_gather_list(local_supported_targets):
            supported_target_counts[str(domain)] = supported_target_counts.get(str(domain), 0) + 1
        globally_supported_targets = {
            domain for domain, count in supported_target_counts.items() if count == max(world_size, 1)
        }

        for domain, candidates in sorted(self._token_gradient_candidates.items()):
            if domain not in globally_supported_targets:
                continue
            for candidate_index, candidate in enumerate(candidates):
                micro_batch = candidate["micro_batch"]
                for token_candidate in candidate.get("tokens", []):
                    token_candidate = dict(token_candidate)
                    token_candidate_id = len(local_metadata)
                    token_candidate["candidate_index"] = candidate_index
                    token_candidate["micro_batch"] = micro_batch
                    token_candidate["loss_scale_factor"] = float(candidate["context"].get("loss_scale_factor", 1.0))
                    token_candidate["on_policy"] = bool(candidate["context"].get("on_policy", False))
                    token_candidate["owner_rank"] = local_rank
                    token_candidate["token_candidate_id"] = token_candidate_id
                    key = (local_rank, token_candidate_id)
                    local_records_by_key[key] = token_candidate
                    local_metadata.append(self._token_gradient_metadata(domain, token_candidate))

        global_metadata = _all_gather_list(local_metadata)
        domains = sorted(
            {
                str(row.get("domain"))
                for row in global_metadata
                if str(row.get("domain")) in globally_supported_targets
            }
        )
        for domain in domains:
            domain_rows: list[dict[str, Any]] = []
            token_metadata = [
                row for row in global_metadata if str(row.get("domain")) == domain
            ]
            selected_samples_by_domain = len(
                self._token_sample_keys(token_metadata)
            )
            total_gap_mass = sum(
                self._token_score_mass_value(row, "gap") for row in token_metadata
            )
            total_gap_abs_mass = sum(float(row.get("gap_abs", 0.0)) for row in token_metadata)
            total_loss_abs_mass = sum(float(row.get("loss_abs", 0.0) or 0.0) for row in token_metadata)
            for selection_name, selection_score_key, selected_metadata in self._token_score_selections(token_metadata):
                if not selected_metadata:
                    continue
                local_selected_tokens = self._local_tokens_for_global_selection(
                    selected_metadata,
                    local_records_by_key,
                    local_rank=local_rank,
                )
                use_sequence_token_replay = bool(
                    self.sequence_masked_target_enabled
                    and self.sequence_masked_target_use_as_primary
                )
                if use_sequence_token_replay:
                    stats = self._recompute_sequence_token_selection_gradient_stats(
                        selected_metadata,
                        target_map=domain_targets,
                        apply_contribution_scale=True,
                    )
                else:
                    stats = self._recompute_token_selection_gradient_stats(
                        local_selected_tokens,
                        target_map=domain_targets,
                        restore_grads=self.token_gradient_strict_grad_restore,
                    )
                token_recompute_attempted = True
                other_domain = self._other_domain(domain, domain_targets)
                own_cos = stats.get(f"{_safe_name(domain)}_cos")
                other_cos = stats.get(f"{_safe_name(other_domain)}_cos") if other_domain is not None else None
                own_projection = stats.get(f"{_safe_name(domain)}_projection_share")
                other_projection = (
                    stats.get(f"{_safe_name(other_domain)}_projection_share")
                    if other_domain is not None
                    else None
                )
                selected_gap_mass = sum(
                    self._token_score_mass_value(row, "gap") for row in selected_metadata
                )
                selected_gap_abs_mass = sum(float(row.get("gap_abs", 0.0)) for row in selected_metadata)
                selected_loss_abs_mass = sum(float(row.get("loss_abs", 0.0) or 0.0) for row in selected_metadata)
                selected_score_mass = sum(
                    self._token_score_mass_value(row, selection_score_key) for row in selected_metadata
                )
                total_score_mass = sum(
                    self._token_score_mass_value(row, selection_score_key) for row in token_metadata
                )
                rank_selected_token_counts = self._rank_token_counts(selected_metadata)
                top_p_full_selection = (
                    self.token_gradient_top_p >= 1.0 - 1e-12
                    and selection_name == self._top_p_selection_name(selection_score_key)
                )
                selected_sample_count = len(self._token_sample_keys(selected_metadata))
                candidate_token_count = len(token_metadata)
                candidate_sample_count = selected_samples_by_domain
                finite_score_token_count = len(
                    [
                        row
                        for row in token_metadata
                        if row.get(selection_score_key) is not None
                        and math.isfinite(float(row.get(selection_score_key, 0.0)))
                    ]
                )
                row = {
                    "step": self.step,
                    "domain": domain,
                    "selection": selection_name,
                    "selection_score": selection_score_key,
                    "selection_scope": "global",
                    "rank": "global",
                    "world_size": world_size,
                    "other_domain": other_domain,
                    "own_domain_cos": own_cos,
                    "other_domain_cos": other_cos,
                    "conflict_to_other": max(0.0, -float(other_cos)) if other_cos is not None else None,
                    "own_projection_share": own_projection,
                    "other_projection_share": other_projection,
                    "selected_token_count": float(len(selected_metadata)),
                    "selected_sample_count": float(selected_sample_count),
                    "selected_rank_count": float(len(rank_selected_token_counts)),
                    "rank_selected_token_counts": rank_selected_token_counts,
                    "local_selected_token_count": float(len(local_selected_tokens)),
                    "global_candidate_token_count": float(candidate_token_count),
                    "global_candidate_sample_count": float(candidate_sample_count),
                    "response_mask_valid_token_count": float(candidate_token_count),
                    "candidate_token_coverage_frac": (
                        candidate_token_count / max(candidate_token_count, 1)
                    ),
                    "finite_score_token_count": float(finite_score_token_count),
                    "finite_score_token_frac": (
                        finite_score_token_count / max(candidate_token_count, 1)
                    ),
                    "global_candidate_gap_mass": total_gap_mass,
                    "global_candidate_gap_abs_mass": total_gap_abs_mass,
                    "global_candidate_loss_abs_mass": total_loss_abs_mass,
                    "global_candidate_score_mass": total_score_mass,
                    "global_candidate_scope": "all_valid_response_tokens",
                    "selected_gap_mass": selected_gap_mass,
                    "selected_gap_mass_frac": selected_gap_mass / total_gap_mass
                    if total_gap_mass > 0.0
                    else 0.0,
                    "selected_gap_mean": selected_gap_mass / max(len(selected_metadata), 1),
                    "selected_gap_abs_mass": selected_gap_abs_mass,
                    "selected_gap_abs_mass_frac": selected_gap_abs_mass / total_gap_abs_mass
                    if total_gap_abs_mass > 0.0
                    else 0.0,
                    "selected_gap_abs_mean": selected_gap_abs_mass / max(len(selected_metadata), 1),
                    "selected_loss_abs_mass": selected_loss_abs_mass,
                    "selected_loss_abs_mass_frac": selected_loss_abs_mass / total_loss_abs_mass
                    if total_loss_abs_mass > 0.0
                    else 0.0,
                    "selected_loss_abs_mean": selected_loss_abs_mass / max(len(selected_metadata), 1),
                    "selected_score_mass": selected_score_mass,
                    "selected_score_mass_frac": selected_score_mass / total_score_mass
                    if total_score_mass > 0.0
                    else 0.0,
                    "selected_score_mean": selected_score_mass / max(len(selected_metadata), 1),
                    **stats,
                }
                if use_sequence_token_replay and top_p_full_selection:
                    no_scale_stats = self._recompute_sequence_token_selection_gradient_stats(
                        selected_metadata,
                        target_map={domain: domain_targets[domain]},
                        apply_contribution_scale=False,
                    )
                    safe_domain = _safe_name(domain)
                    row["token_sequence_scale_no_extra_available"] = no_scale_stats.get(
                        "token_grad_available"
                    )
                    row["token_sequence_scale_no_extra_cos"] = no_scale_stats.get(
                        f"{safe_domain}_cos"
                    )
                    row["token_sequence_scale_no_extra_projection_share"] = no_scale_stats.get(
                        f"{safe_domain}_projection_share"
                    )
                    row["token_sequence_scale_no_extra_norm_ratio"] = no_scale_stats.get(
                        f"{safe_domain}_norm_ratio"
                    )
                if top_p_full_selection:
                    self._annotate_token_closure_row(
                        row,
                        domain=domain,
                        domain_targets=domain_targets,
                        selected_token_count=len(selected_metadata),
                        candidate_token_count=candidate_token_count,
                        selected_sample_count=selected_sample_count,
                        candidate_sample_count=candidate_sample_count,
                    )
                domain_rows.append(row)
            rows.extend(domain_rows)
            metrics.update(self._summarize_token_gradient_rows(domain, domain_rows))
        if token_recompute_attempted:
            _restore_parameter_grads_from_snapshot(
                parameters_for_final_restore,
                final_grad_snapshot,
                grad_dtypes=final_grad_dtypes,
            )
        if rows and local_rank == 0:
            _write_jsonl_rows(self.output_dir, "token_grad_metrics.jsonl", rows)
        total_seconds = _all_reduce_values_max([time.perf_counter() - started_at])[0]
        metrics.update(self._summarize_global_token_gradient_cost(rows, total_seconds))
        return metrics

    def _annotate_token_closure_row(
        self,
        row: dict[str, Any],
        *,
        domain: str,
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]],
        selected_token_count: int,
        candidate_token_count: int,
        selected_sample_count: int,
        candidate_sample_count: int,
    ) -> None:
        safe_domain = _safe_name(domain)
        row["closure_candidate_token_frac"] = (
            selected_token_count / candidate_token_count
            if candidate_token_count > 0
            else 0.0
        )
        row["closure_candidate_sample_frac"] = (
            selected_sample_count / candidate_sample_count
            if candidate_sample_count > 0
            else 0.0
        )
        row["closure_selected_all_tokens"] = float(
            selected_token_count == candidate_token_count and candidate_token_count > 0
        )
        row["closure_selected_all_samples"] = float(
            selected_sample_count == candidate_sample_count and candidate_sample_count > 0
        )
        own_projection = row.get(f"{safe_domain}_projection_share")
        own_cos = row.get(f"{safe_domain}_cos")
        if own_projection is not None:
            row["closure_projection_share_error"] = abs(float(own_projection) - 1.0)
        if own_cos is not None:
            row["closure_cosine_error"] = abs(float(own_cos) - 1.0)
        target = domain_targets.get(domain)
        token_norm = row.get("token_grad_norm")
        if target is None or token_norm is None:
            return
        target_norm = float(max(target[1], 0.0) ** 0.5)
        if target_norm <= 0.0:
            return
        norm_ratio = float(token_norm) / target_norm
        row["closure_norm_ratio"] = norm_ratio
        row["closure_norm_ratio_error"] = abs(norm_ratio - 1.0)

    def _token_gradient_metadata(self, domain: str, token: dict[str, Any]) -> dict[str, Any]:
        metadata_keys = (
            "sample_id",
            "sample_index",
            "position",
            "rank_in_sample",
            "gap",
            "gap_signed",
            "gap_abs",
            "teacher_logp",
            "student_logp",
            "loss_signed",
            "loss_abs",
            "loss_raw_signed",
            "loss_raw_abs",
            "loss_contribution_scale",
            "loss_score_source",
            "effective_tokens",
            "token_id",
            "original_sample_index",
            "source_micro_batch_index",
            "owner_rank",
            "token_candidate_id",
        )
        metadata = {"domain": domain}
        for key in metadata_keys:
            if key in token:
                metadata[key] = token[key]
        return _json_safe(metadata)

    def _local_tokens_for_global_selection(
        self,
        selected_metadata: list[dict[str, Any]],
        local_records_by_key: dict[tuple[int, int], dict[str, Any]],
        *,
        local_rank: int,
    ) -> list[dict[str, Any]]:
        local_tokens: list[dict[str, Any]] = []
        for row in selected_metadata:
            owner_rank = int(row.get("owner_rank", -1))
            if owner_rank != local_rank:
                continue
            key = (owner_rank, int(row.get("token_candidate_id", -1)))
            token = local_records_by_key.get(key)
            if token is not None:
                local_tokens.append(token)
        return local_tokens

    def _rank_token_counts(self, selected_metadata: list[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in selected_metadata:
            rank = str(int(row.get("owner_rank", -1)))
            counts[rank] = counts.get(rank, 0) + 1
        return counts

    def _token_sample_keys(self, token_metadata: list[dict[str, Any]]) -> set[tuple[str, ...]]:
        keys: set[tuple[str, ...]] = set()
        for row in token_metadata:
            owner_rank = str(row.get("owner_rank", -1))
            sample_id = str(row.get("sample_id", ""))
            source_micro_batch_index = row.get("source_micro_batch_index")
            original_sample_index = row.get("original_sample_index", row.get("sample_index"))
            if source_micro_batch_index is not None and original_sample_index is not None:
                keys.add((owner_rank, str(source_micro_batch_index), str(original_sample_index), sample_id))
            elif sample_id:
                keys.add((owner_rank, sample_id))
            else:
                keys.add((owner_rank, str(row.get("token_candidate_id", -1))))
        return keys

    def _token_score_selections(
        self,
        token_records: list[dict[str, Any]],
    ) -> list[tuple[str, str, list[dict[str, Any]]]]:
        if not token_records:
            return []
        selections: list[tuple[str, str, list[dict[str, Any]]]] = []
        score_keys: list[str] = []
        if self.token_gradient_gap_selection_enabled:
            score_keys.append("gap")
        if self.token_gradient_gap_abs_selection_enabled:
            score_keys.append("gap_abs")
        if self.token_gradient_loss_abs_selection_enabled:
            score_keys.append("loss_abs")
        for score_key in score_keys:
            scored_records = [
                row
                for row in token_records
                if row.get(score_key) is not None and math.isfinite(float(row.get(score_key, 0.0)))
            ]
            if not scored_records:
                continue
            sorted_records = sorted(
                scored_records,
                key=lambda row: (
                    -float(row.get(score_key, 0.0)),
                    int(row.get("owner_rank", 0)),
                    int(row.get("token_candidate_id", 0)),
                ),
            )
            top_k = sorted_records[: min(self.token_gradient_top_k, len(sorted_records))]
            total_mass = sum(self._token_score_mass_value(row, score_key) for row in sorted_records)
            top_p_selection: list[dict[str, Any]] = []
            if self.token_gradient_top_p >= 1.0 - 1e-12:
                top_p_selection = list(sorted_records)
            elif total_mass > 0.0:
                top_p_threshold = total_mass * self.token_gradient_top_p
                running = 0.0
                for row in sorted_records:
                    top_p_selection.append(row)
                    running += self._token_score_mass_value(row, score_key)
                    if running >= top_p_threshold:
                        break
            else:
                top_p_selection = sorted_records[:1]
            selections.append((f"top{self.token_gradient_top_k}_{score_key}", score_key, top_k))
            selections.append((self._top_p_selection_name(score_key), score_key, top_p_selection))
        return selections

    def _token_score_mass_value(self, row: dict[str, Any], score_key: str) -> float:
        value = float(row.get(score_key, 0.0) or 0.0)
        if score_key == "gap":
            return max(0.0, value)
        return value

    def _gap_abs_token_selections(
        self,
        token_records: list[dict[str, Any]],
    ) -> list[tuple[str, list[dict[str, Any]]]]:
        return [
            (selection, rows)
            for selection, score_key, rows in self._token_score_selections(token_records)
            if score_key == "gap_abs"
        ]

    def _top_p_selection_name(self, score_key: str) -> str:
        percent = self.token_gradient_top_p * 100.0
        rounded = round(percent)
        if abs(percent - rounded) < 1e-6:
            percent_label = str(int(rounded))
        else:
            percent_label = f"{percent:.2f}".rstrip("0").rstrip(".").replace(".", "p")
        return f"topp{percent_label}_{score_key}_mass"

    def _select_token_gradient_candidates(
        self,
        micro_batch: DataProto,
        *,
        domain: str,
        fallback_prefix: str | None = None,
        on_policy: bool = False,
        loss_scale_factor: float = 1.0,
    ) -> list[dict[str, Any]]:
        if micro_batch.batch is None:
            return []
        model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
        if "response_mask" not in model_inputs:
            return []
        response_mask = model_inputs["response_mask"].detach().float().cpu()
        policy_loss_cfg = _cfg_get(self.actor.config, "policy_loss", {})
        topk_distill_active = uses_topk_distill_loss(policy_loss_cfg)
        has_gap_inputs = "old_log_probs" in model_inputs and "math_teacher_log_prob" in model_inputs
        if not has_gap_inputs and not (
            topk_distill_active and self.token_gradient_loss_abs_selection_enabled
        ):
            return []

        old_log_probs = None
        selected_teacher = None
        if has_gap_inputs:
            old_log_probs = model_inputs["old_log_probs"].detach().float().cpu()
            try:
                selected_teacher = _selected_teacher_log_prob_from_inputs(
                    model_inputs,
                    policy_loss_cfg,
                ).detach().float().cpu()
            except Exception:
                return []
            if tuple(old_log_probs.shape) != tuple(response_mask.shape):
                return []
            if tuple(selected_teacher.shape) != tuple(response_mask.shape):
                return []

        labels = _labels_from_inputs(model_inputs, int(response_mask.shape[0]))
        sample_ids = _sample_ids(micro_batch, self.step, fallback_prefix=fallback_prefix)
        token_ids = _response_token_id_matrix_from_inputs(model_inputs, response_mask)
        loss_scores = None
        loss_score_source = "disabled"
        loss_agg_mode = str(_cfg_get(self.actor.config, "loss_agg_mode", "token-mean"))
        if self.token_gradient_loss_abs_selection_enabled:
            loss_scores, loss_score_source = _actor_micro_batch_token_loss_scores(
                self.actor,
                micro_batch,
                on_policy=on_policy,
            )
        if loss_scores is not None:
            loss_scores = loss_scores.to(dtype=torch.float32, device=response_mask.device)
        if not has_gap_inputs and loss_scores is None:
            return []
        rows: list[dict[str, Any]] = []
        for sample_idx, label in enumerate(labels):
            if label != domain:
                continue
            valid_positions = torch.nonzero(response_mask[sample_idx] > 0, as_tuple=False).reshape(-1)
            if valid_positions.numel() == 0:
                continue
            gap_signed = None
            gap_abs = None
            valid_scores = None
            if old_log_probs is not None and selected_teacher is not None:
                gap_signed = (selected_teacher[sample_idx] - old_log_probs[sample_idx]) * response_mask[sample_idx]
                gap_abs = gap_signed.abs()
                valid_scores = gap_abs[valid_positions]
            if (
                valid_scores is None
                and loss_scores is not None
                and tuple(loss_scores.shape) == tuple(response_mask.shape)
            ):
                valid_scores = loss_scores[sample_idx].abs()[valid_positions]
            if valid_scores is None:
                valid_scores = response_mask[sample_idx][valid_positions]
            sorted_offsets = torch.argsort(valid_scores, descending=True)
            for rank, offset in enumerate(sorted_offsets.tolist(), start=1):
                position = int(valid_positions[offset].item())
                row: dict[str, Any] = {
                    "sample_id": sample_ids[sample_idx],
                    "sample_index": int(sample_idx),
                    "position": position,
                    "rank_in_sample": rank,
                    "effective_tokens": float(response_mask[sample_idx].sum().detach().cpu().item()),
                }
                if gap_signed is not None and gap_abs is not None and old_log_probs is not None and selected_teacher is not None:
                    gap_abs_value = float(gap_abs[position].detach().cpu().item())
                    gap_signed_value = float(gap_signed[position].detach().cpu().item())
                    row.update(
                        {
                            "gap": gap_signed_value,
                            "gap_signed": gap_signed_value,
                            "gap_abs": gap_abs_value,
                            "gap_score_source": "teacher_student_logprob_proxy",
                            "teacher_logp": float(selected_teacher[sample_idx, position].detach().cpu().item()),
                            "student_logp": float(old_log_probs[sample_idx, position].detach().cpu().item()),
                        }
                    )
                if loss_scores is not None and tuple(loss_scores.shape) == tuple(response_mask.shape):
                    raw_loss_signed_value = float(loss_scores[sample_idx, position].detach().cpu().item())
                    contribution_scale = float(loss_scale_factor) * _token_contribution_scale(
                        response_mask,
                        sample_idx,
                        loss_agg_mode,
                    )
                    loss_signed_value = raw_loss_signed_value * contribution_scale
                    row["loss_signed"] = loss_signed_value
                    row["loss_abs"] = abs(loss_signed_value)
                    row["loss_raw_signed"] = raw_loss_signed_value
                    row["loss_raw_abs"] = abs(raw_loss_signed_value)
                    row["loss_contribution_scale"] = contribution_scale
                    row["loss_score_source"] = f"{loss_score_source}_contribution_scaled"
                if token_ids is not None:
                    row["token_id"] = int(token_ids[sample_idx, position].detach().cpu().item())
                rows.append(row)
        return rows

    def _other_domain(
        self,
        domain: str,
        domain_targets: dict[str, tuple[tuple[torch.Tensor, ...], float]],
    ) -> str | None:
        for candidate in self.domains:
            if candidate != domain and candidate in domain_targets:
                return candidate
        for candidate in sorted(domain_targets):
            if candidate != domain:
                return candidate
        return None

    def _recompute_token_selection_gradient_stats(
        self,
        selected_tokens: list[dict[str, Any]],
        *,
        target_map: dict[str, tuple[tuple[torch.Tensor, ...], float]],
        restore_grads: bool = True,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        parameters = _trainable_parameters(self.actor)
        grad_dtypes = _parameter_grad_dtypes(parameters) if selected_tokens else tuple()
        grad_snapshot = (
            _snapshot_parameter_grads(parameters)
            if selected_tokens and self.token_gradient_strict_grad_restore and restore_grads
            else None
        )
        pre_diff = (
            _parameter_grad_target_diff_stats(parameters, target_map)
            if selected_tokens and restore_grads
            else {}
        )
        post_diff: dict[str, float] = {}
        original_diff: dict[str, float] | None = None
        local_norm_sq: float | None = 0.0
        local_dots: dict[str, float] | None = {domain: 0.0 for domain in target_map}
        non_none_grad_count = 0
        autograd_error: str | None = None
        autograd_started_at = time.perf_counter()
        actor_config = getattr(self.actor, "config", {})
        loss_agg_mode = str(_cfg_get(actor_config, "loss_agg_mode", "token-mean"))
        replica_count = _gradient_replica_count(self.actor)
        use_backward_recompute = bool(self.token_gradient_backward_recompute_enabled)
        gradients_synced_by_backward = bool(
            use_backward_recompute
            and self.token_gradient_backward_sync_enabled
            and replica_count > 1
        )
        gradients_are_optimizer_space = bool(
            replica_count <= 1
            or gradients_synced_by_backward
        )

        grouped: dict[int, dict[str, Any]] = {}
        for token in selected_tokens:
            group_key = int(token["candidate_index"])
            group = grouped.setdefault(
                group_key,
                {
                    "micro_batch": token["micro_batch"],
                    "loss_scale_factor": float(token.get("loss_scale_factor", 1.0)),
                    "on_policy": bool(token.get("on_policy", False)),
                    "positions": [],
                },
                )
            group["positions"].append((int(token["sample_index"]), int(token["position"])))

        used_micro_batches: list[DataProto] = []
        gradients: tuple[torch.Tensor | None, ...] = tuple(None for _ in parameters)
        if selected_tokens:
            try:
                accumulated_gradients: list[torch.Tensor | None] = [None for _ in parameters]
                if use_backward_recompute:
                    _clear_parameter_grads(parameters)
                for group in grouped.values():
                    micro_batch = group["micro_batch"]
                    used_micro_batches.append(micro_batch)
                    token_mask = torch.zeros_like(micro_batch.batch["response_mask"], dtype=torch.float32)
                    for sample_idx, position in group["positions"]:
                        token_mask[sample_idx, position] = 1.0
                    contribution_scale = _token_mask_contribution_scale(
                        micro_batch.batch["response_mask"],
                        token_mask,
                        loss_agg_mode,
                    )
                    if contribution_scale <= 0.0:
                        continue
                    recompute_sync_context = (
                        nullcontext()
                        if not use_backward_recompute or self.token_gradient_backward_sync_enabled
                        else _actor_no_sync_context(self.actor)
                    )
                    with recompute_sync_context:
                        loss = _actor_micro_batch_loss(
                            self.actor,
                            micro_batch,
                            loss_scale_factor=float(group["loss_scale_factor"]) * contribution_scale,
                            on_policy=bool(group["on_policy"]),
                            safe_logprob_backward=not use_backward_recompute,
                            response_mask_override=token_mask,
                        )
                        if use_backward_recompute:
                            loss.backward()
                            del loss
                            continue
                        group_gradients = torch.autograd.grad(
                            loss,
                            parameters,
                            retain_graph=False,
                            allow_unused=True,
                        )
                    for idx, gradient in enumerate(group_gradients):
                        if gradient is None:
                            continue
                        detached_gradient = gradient.detach()
                        if accumulated_gradients[idx] is None:
                            accumulated_gradients[idx] = detached_gradient.clone()
                        else:
                            accumulated_gradients[idx] = accumulated_gradients[idx] + detached_gradient
                    del group_gradients, loss
                if use_backward_recompute:
                    _finalize_fsdp_after_auxiliary_backward(self.actor)
                    gradients = tuple(parameter.grad for parameter in parameters)
                else:
                    gradients = tuple(accumulated_gradients)
                non_none_grad_count = sum(gradient is not None for gradient in gradients)
                local_norm_sq, local_dots = self._grad_multi_target_stats_from_tensors(gradients, target_map)
                del gradients
            except Exception as exc:
                autograd_error = type(exc).__name__
            finally:
                _finalize_fsdp_after_auxiliary_backward(self.actor)
                if restore_grads:
                    if grad_snapshot is not None:
                        _restore_parameter_grads_from_snapshot(parameters, grad_snapshot, grad_dtypes=grad_dtypes)
                        original_diff = _parameter_grad_snapshot_diff_stats(parameters, grad_snapshot)
                    else:
                        _restore_parameter_grads_from_targets(parameters, target_map, grad_dtypes=grad_dtypes)
                    post_diff = _parameter_grad_target_diff_stats(parameters, target_map)
                for micro_batch in used_micro_batches:
                    try:
                        micro_batch.to("cpu")
                    except Exception:
                        pass
        autograd_seconds = time.perf_counter() - autograd_started_at
        restore_stats = {
            "token_grad_restore_pre_target_rel_l2": pre_diff.get("rel_l2", 0.0),
            "token_grad_restore_pre_target_max_abs": pre_diff.get("max_abs", 0.0),
            "token_grad_restore_post_target_rel_l2": post_diff.get("rel_l2", 0.0),
            "token_grad_restore_post_target_max_abs": post_diff.get("max_abs", 0.0),
            "token_grad_restore_target_norm": pre_diff.get("target_norm", 0.0),
        }
        if original_diff is not None:
            restore_stats.update(
                {
                    "token_grad_restore_original_rel_l2": original_diff["rel_l2"],
                    "token_grad_restore_original_max_abs": original_diff["max_abs"],
                    "token_grad_restore_original_norm": original_diff["snapshot_norm"],
                }
            )
        else:
            restore_stats.update(
                {
                    "token_grad_restore_original_rel_l2": 0.0,
                    "token_grad_restore_original_max_abs": 0.0,
                    "token_grad_restore_original_norm": 0.0,
                }
            )
        local_error = autograd_error
        if local_norm_sq is None or local_dots is None:
            local_error = autograd_error or "parameter_target_mismatch"
            local_norm_sq = 0.0
            local_dots = {domain: 0.0 for domain in target_map}
        if selected_tokens and non_none_grad_count == 0 and local_error is None:
            local_error = "all_parameters_disconnected"

        local_param_count = len(parameters) if selected_tokens else 0

        target_domains = sorted(target_map)
        reduced_values = _all_reduce_values_sum(
            [float(max(local_norm_sq, 0.0)), float(non_none_grad_count), float(local_param_count)]
            + [float(local_dots.get(domain, 0.0)) for domain in target_domains]
        )
        candidate_scale = 1.0
        count_scale = 1.0
        if replica_count > 1:
            if gradients_are_optimizer_space:
                count_scale = 1.0 / float(replica_count)
            else:
                candidate_scale = 1.0 / float(replica_count)
        norm_sq = max(reduced_values[0], 0.0)
        dots = {
            domain: reduced_values[3 + idx]
            for idx, domain in enumerate(target_domains)
        }
        if not gradients_are_optimizer_space:
            norm_sq *= candidate_scale * candidate_scale
            dots = {domain: dot * candidate_scale for domain, dot in dots.items()}
        global_non_none_grad_count = reduced_values[1] * count_scale
        global_param_count = reduced_values[2] * count_scale
        global_none_grad_count = max(global_param_count - global_non_none_grad_count, 0.0)
        max_keys = [
            "token_grad_seconds",
            "token_grad_autograd_seconds",
            "token_grad_restore_pre_target_rel_l2",
            "token_grad_restore_pre_target_max_abs",
            "token_grad_restore_post_target_rel_l2",
            "token_grad_restore_post_target_max_abs",
            "token_grad_restore_target_norm",
            "token_grad_restore_original_rel_l2",
            "token_grad_restore_original_max_abs",
            "token_grad_restore_original_norm",
        ]
        local_timing_and_restore = {
            "token_grad_seconds": time.perf_counter() - started_at,
            "token_grad_autograd_seconds": autograd_seconds,
            **restore_stats,
        }
        max_values = _all_reduce_values_max([float(local_timing_and_restore.get(key, 0.0)) for key in max_keys])
        reduced_max = {key: max_values[idx] for idx, key in enumerate(max_keys)}
        global_errors = sorted(set(str(error) for error in _all_gather_list([local_error] if local_error else [])))

        token_norm = norm_sq**0.5
        available = not global_errors and global_non_none_grad_count > 0 and token_norm > 0.0
        stats: dict[str, Any] = {
            "token_grad_available": float(available),
            "token_grad_norm": token_norm if available else None,
            "token_grad_non_none_grad_count": float(global_non_none_grad_count),
            "token_grad_param_count": float(global_param_count),
            "token_grad_none_grad_count": float(global_none_grad_count),
            "token_grad_autograd_error": ";".join(global_errors) if global_errors else None,
            "token_grad_backward_fallback_seconds": 0.0,
            "token_grad_backward_fallback_used": 0.0,
            "token_grad_backward_recompute_used": float(self.token_gradient_backward_recompute_enabled),
            "token_grad_backward_sync_used": float(self.token_gradient_backward_sync_enabled),
            "token_grad_backward_sync_replica_average_used": float(gradients_synced_by_backward),
            "token_grad_optimizer_space_used": float(gradients_are_optimizer_space),
            "token_grad_read_after_finalize": float(use_backward_recompute),
            "token_grad_pre_finalize_read_disabled": float(use_backward_recompute),
            "token_grad_candidate_scale": candidate_scale,
            "token_grad_replica_scale_applied": float(
                replica_count > 1 and candidate_scale != 1.0
            ),
            "token_grad_replica_count": float(replica_count),
            **reduced_max,
        }
        for target_domain, dot in dots.items():
            _chunks, target_norm_sq = target_map[target_domain]
            target_norm = target_norm_sq**0.5
            safe_domain = _safe_name(target_domain)
            stats[f"{safe_domain}_cos"] = dot / (token_norm * target_norm) if available and target_norm > 0.0 else None
            stats[f"{safe_domain}_projection_share"] = (
                dot / target_norm_sq if available and target_norm_sq > 0.0 else None
            )
            stats[f"{safe_domain}_norm_ratio"] = (
                token_norm / target_norm if available and target_norm > 0.0 else None
            )
        return stats

    def _recompute_sequence_token_selection_gradient_stats(
        self,
        selected_metadata: list[dict[str, Any]],
        *,
        target_map: dict[str, tuple[tuple[torch.Tensor, ...], float]],
        apply_contribution_scale: bool = True,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        parameters = _trainable_parameters(self.actor)
        target_spec = {
            "type": "token_selection",
            "tokens": selected_metadata,
            "apply_token_mask_contribution_scale": bool(apply_contribution_scale),
        }
        sequence_metrics, chunks, norm_sq = self._recompute_masked_schedule_target(
            target_spec,
            storage_dtype="float32",
        )
        token_mask_sum = float(
            sequence_metrics.get("global/audit/sequence_target_token_selection_token_mask_sum", 0.0)
            or 0.0
        )
        effective_loss_scale_sum = float(
            sequence_metrics.get(
                "global/audit/sequence_target_token_selection_effective_loss_scale_sum",
                0.0,
            )
            or 0.0
        )
        available = bool(chunks and norm_sq > 0.0)
        target_domains = sorted(target_map)
        token_norm = max(norm_sq, 0.0) ** 0.5
        stats: dict[str, Any] = {
            "token_grad_available": float(available),
            "token_grad_norm": token_norm if available else None,
            "token_grad_non_none_grad_count": float(len(parameters) if available else 0),
            "token_grad_param_count": float(len(parameters)),
            "token_grad_none_grad_count": float(0 if available else len(parameters)),
            "token_grad_autograd_error": None if available else "sequence_token_target_unavailable",
            "token_grad_backward_fallback_seconds": 0.0,
            "token_grad_backward_fallback_used": 0.0,
            "token_grad_backward_recompute_used": 1.0,
            "token_grad_backward_sync_used": 1.0,
            "token_grad_backward_sync_replica_average_used": 1.0,
            "token_grad_optimizer_space_used": 1.0,
            "token_grad_read_after_finalize": 1.0,
            "token_grad_pre_finalize_read_disabled": 1.0,
            "token_grad_candidate_scale": 1.0,
            "token_grad_replica_scale_applied": 0.0,
            "token_grad_replica_count": float(_gradient_replica_count(self.actor)),
            "token_grad_seconds": time.perf_counter() - started_at,
            "token_grad_autograd_seconds": sequence_metrics.get(
                "global/full_grad_cost/sequence_target_token_selection_seconds",
                0.0,
            ),
            "token_grad_restore_pre_target_rel_l2": 0.0,
            "token_grad_restore_pre_target_max_abs": 0.0,
            "token_grad_restore_post_target_rel_l2": 0.0,
            "token_grad_restore_post_target_max_abs": 0.0,
            "token_grad_restore_target_norm": 0.0,
            "token_grad_restore_original_rel_l2": 0.0,
            "token_grad_restore_original_max_abs": 0.0,
            "token_grad_restore_original_norm": 0.0,
            "token_grad_sequence_replay_used": 1.0,
            "token_grad_sequence_apply_contribution_scale": float(apply_contribution_scale),
            "token_grad_sequence_token_mask_sum": token_mask_sum,
            "token_grad_sequence_effective_loss_scale_sum": effective_loss_scale_sum,
        }
        if not available:
            for target_domain in target_domains:
                safe_domain = _safe_name(target_domain)
                stats[f"{safe_domain}_cos"] = None
                stats[f"{safe_domain}_projection_share"] = None
                stats[f"{safe_domain}_norm_ratio"] = None
            return stats

        for target_domain in target_domains:
            target_chunks, target_norm_sq = target_map[target_domain]
            dot = self._target_chunks_dot(chunks, target_chunks)
            safe_domain = _safe_name(target_domain)
            if dot is None:
                stats[f"{safe_domain}_cos"] = None
                stats[f"{safe_domain}_projection_share"] = None
                stats[f"{safe_domain}_norm_ratio"] = None
                continue
            target_norm = max(target_norm_sq, 0.0) ** 0.5
            stats[f"{safe_domain}_cos"] = (
                dot / (token_norm * target_norm)
                if token_norm > 0.0 and target_norm > 0.0
                else None
            )
            stats[f"{safe_domain}_projection_share"] = (
                dot / target_norm_sq if target_norm_sq > 0.0 else None
            )
            stats[f"{safe_domain}_norm_ratio"] = (
                token_norm / target_norm if target_norm > 0.0 else None
            )
        return stats

    def _grad_multi_target_stats_from_tensors(
        self,
        gradients: tuple[torch.Tensor | None, ...],
        target_map: dict[str, tuple[tuple[torch.Tensor, ...], float]],
    ) -> tuple[float | None, dict[str, float] | None]:
        local_norm_sq = 0.0
        local_dots = {domain: 0.0 for domain in target_map}
        for param_idx, gradient in enumerate(gradients):
            if gradient is None:
                continue
            gradient_gpu = gradient.detach().reshape(-1).float()
            gradient_norm_sq = _chunked_vector_dot(gradient_gpu, gradient_gpu)
            if gradient_norm_sq is None:
                return None, None
            local_norm_sq += gradient_norm_sq
            for domain, (target_chunks, _target_norm_sq) in target_map.items():
                if param_idx >= len(target_chunks):
                    return None, None
                target = target_chunks[param_idx]
                if gradient_gpu.numel() != target.numel():
                    return None, None
                target_gpu = target.reshape(-1).to(device=gradient_gpu.device, dtype=torch.float32)
                gradient_target_dot = _chunked_vector_dot(gradient_gpu, target_gpu)
                if gradient_target_dot is None:
                    return None, None
                local_dots[domain] += gradient_target_dot
        return local_norm_sq, local_dots

    def _summarize_token_gradient_rows(
        self,
        domain: str,
        rows: list[dict[str, Any]],
    ) -> dict[str, float]:
        safe_domain = _safe_name(domain)
        finite_norms = _finite_values([row.get("token_grad_norm") for row in rows])
        other_cos: list[float] = []
        own_projection: list[float] = []
        other_projection: list[float] = []
        seconds = _finite_values([row.get("token_grad_seconds") for row in rows])
        autograd_seconds = _finite_values([row.get("token_grad_autograd_seconds") for row in rows])
        fallback_seconds = _finite_values([row.get("token_grad_backward_fallback_seconds") for row in rows])
        fallback_used = _finite_values([row.get("token_grad_backward_fallback_used") for row in rows])
        availability = _finite_values([row.get("token_grad_available") for row in rows])
        pre_restore_rel_l2 = _finite_values([row.get("token_grad_restore_pre_target_rel_l2") for row in rows])
        post_restore_rel_l2 = _finite_values([row.get("token_grad_restore_post_target_rel_l2") for row in rows])
        pre_restore_max_abs = _finite_values([row.get("token_grad_restore_pre_target_max_abs") for row in rows])
        post_restore_max_abs = _finite_values([row.get("token_grad_restore_post_target_max_abs") for row in rows])
        original_restore_rel_l2 = _finite_values([row.get("token_grad_restore_original_rel_l2") for row in rows])
        original_restore_max_abs = _finite_values([row.get("token_grad_restore_original_max_abs") for row in rows])
        for row in rows:
            other_domain = row.get("other_domain")
            if other_domain is not None:
                other_cos.extend(_finite_values([row.get(f"{_safe_name(other_domain)}_cos")]))
                other_projection.extend(_finite_values([row.get(f"{_safe_name(other_domain)}_projection_share")]))
            own_projection.extend(_finite_values([row.get(f"{safe_domain}_projection_share")]))
        selected_token_total = sum(
            float(row.get("selected_token_count", 1.0) or 0.0) for row in rows
        )
        selected_sample_total = sum(
            float(row.get("selected_sample_count", 0.0) or 0.0) for row in rows
        )
        global_candidate_token_count = max(
            _finite_values([row.get("global_candidate_token_count") for row in rows]) or [0.0]
        )
        global_candidate_sample_count = max(
            _finite_values([row.get("global_candidate_sample_count") for row in rows]) or [0.0]
        )
        global_candidate_gap_mass = max(
            _finite_values([row.get("global_candidate_gap_mass") for row in rows]) or [0.0]
        )
        global_candidate_gap_abs_mass = max(
            _finite_values([row.get("global_candidate_gap_abs_mass") for row in rows]) or [0.0]
        )
        global_candidate_loss_abs_mass = max(
            _finite_values([row.get("global_candidate_loss_abs_mass") for row in rows]) or [0.0]
        )
        response_mask_valid_token_count = max(
            _finite_values([row.get("response_mask_valid_token_count") for row in rows]) or [0.0]
        )
        candidate_token_coverage_frac = max(
            _finite_values([row.get("candidate_token_coverage_frac") for row in rows]) or [0.0]
        )
        finite_score_token_count = max(
            _finite_values([row.get("finite_score_token_count") for row in rows]) or [0.0]
        )
        finite_score_token_frac = max(
            _finite_values([row.get("finite_score_token_frac") for row in rows]) or [0.0]
        )
        metrics: dict[str, float] = {
            f"{safe_domain}/token_grad/selected_sample_count": float(selected_sample_total),
            f"{safe_domain}/token_grad/selected_token_count": float(selected_token_total),
            f"{safe_domain}/token_grad/global_candidate_sample_count": float(global_candidate_sample_count),
            f"{safe_domain}/token_grad/global_candidate_token_count": float(global_candidate_token_count),
            f"{safe_domain}/token_metadata/response_mask_valid_token_count": float(
                response_mask_valid_token_count
            ),
            f"{safe_domain}/token_metadata/candidate_token_count": float(global_candidate_token_count),
            f"{safe_domain}/token_metadata/coverage_frac": float(candidate_token_coverage_frac),
            f"{safe_domain}/token_metadata/finite_score_token_count": float(finite_score_token_count),
            f"{safe_domain}/token_metadata/finite_score_token_frac": float(finite_score_token_frac),
            f"{safe_domain}/token_grad/global_candidate_gap_mass": float(global_candidate_gap_mass),
            f"{safe_domain}/token_grad/global_candidate_gap_abs_mass": float(global_candidate_gap_abs_mass),
            f"{safe_domain}/token_grad/global_candidate_loss_abs_mass": float(global_candidate_loss_abs_mass),
        }
        for row in rows:
            selection = row.get("selection")
            if selection is None:
                continue
            selection_key = _safe_name(selection)
            own_cos = row.get(f"{safe_domain}_cos")
            own_projection_value = row.get(f"{safe_domain}_projection_share")
            if own_cos is not None:
                metrics[f"{safe_domain}/token_grad/{selection_key}_cos_to_domain"] = float(own_cos)
            if own_projection_value is not None:
                metrics[
                    f"{safe_domain}/token_grad_contribution/{selection_key}_projection_share"
                ] = float(own_projection_value)
            no_extra_available = row.get("token_sequence_scale_no_extra_available")
            no_extra_cos = row.get("token_sequence_scale_no_extra_cos")
            no_extra_projection = row.get("token_sequence_scale_no_extra_projection_share")
            no_extra_norm_ratio = row.get("token_sequence_scale_no_extra_norm_ratio")
            if no_extra_available is not None:
                metrics[
                    f"{safe_domain}/token_sequence_scale/no_extra_scale/"
                    f"{selection_key}_available"
                ] = float(no_extra_available)
            if no_extra_cos is not None:
                metrics[
                    f"{safe_domain}/token_sequence_scale/no_extra_scale/"
                    f"{selection_key}_cosine"
                ] = float(no_extra_cos)
            if no_extra_projection is not None:
                metrics[
                    f"{safe_domain}/token_sequence_scale/no_extra_scale/"
                    f"{selection_key}_projection_share"
                ] = float(no_extra_projection)
            if no_extra_norm_ratio is not None:
                metrics[
                    f"{safe_domain}/token_sequence_scale/no_extra_scale/"
                    f"{selection_key}_norm_ratio"
                ] = float(no_extra_norm_ratio)
            metrics[f"{safe_domain}/token_grad/{selection_key}_selected_token_count"] = float(
                row.get("selected_token_count", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_non_none_grad_count"] = float(
                row.get("token_grad_non_none_grad_count", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_none_grad_count"] = float(
                row.get("token_grad_none_grad_count", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_param_count"] = float(
                row.get("token_grad_param_count", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_selected_sample_count"] = float(
                row.get("selected_sample_count", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_gap_mass"] = float(
                row.get("selected_gap_mass", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_gap_mass_frac"] = float(
                row.get("selected_gap_mass_frac", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_gap_abs_mass"] = float(
                row.get("selected_gap_abs_mass", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_gap_abs_mass_frac"] = float(
                row.get("selected_gap_abs_mass_frac", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_loss_abs_mass"] = float(
                row.get("selected_loss_abs_mass", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_loss_abs_mass_frac"] = float(
                row.get("selected_loss_abs_mass_frac", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_score_mass"] = float(
                row.get("selected_score_mass", 0.0) or 0.0
            )
            metrics[f"{safe_domain}/token_grad/{selection_key}_score_mass_frac"] = float(
                row.get("selected_score_mass_frac", 0.0) or 0.0
            )
            closure_prefix = f"{safe_domain}/token_grad_closure/{selection_key}"
            for metric_key in (
                "closure_candidate_token_frac",
                "closure_candidate_sample_frac",
                "closure_selected_all_tokens",
                "closure_selected_all_samples",
                "closure_projection_share_error",
                "closure_cosine_error",
                "closure_norm_ratio",
                "closure_norm_ratio_error",
            ):
                value = row.get(metric_key)
                if value is not None:
                    metrics[f"{closure_prefix}_{metric_key.removeprefix('closure_')}"] = float(value)
            if (
                float(row.get("closure_selected_all_tokens", 0.0) or 0.0) > 0.5
                and float(row.get("closure_selected_all_samples", 0.0) or 0.0) > 0.5
            ):
                own_norm_ratio = row.get(f"{safe_domain}_norm_ratio")
                if own_cos is not None:
                    metrics[f"{safe_domain}/token_sequence_closure/all_tokens_vs_domain_cosine"] = (
                        float(own_cos)
                    )
                if own_projection_value is not None:
                    metrics[
                        f"{safe_domain}/token_sequence_closure/"
                        "all_tokens_vs_domain_projection_share"
                    ] = float(own_projection_value)
                    metrics[
                        f"{safe_domain}/token_sequence_closure/"
                        "all_tokens_vs_domain_projection_share_error"
                    ] = abs(float(own_projection_value) - 1.0)
                if own_norm_ratio is not None:
                    metrics[f"{safe_domain}/token_sequence_closure/all_tokens_vs_domain_norm_ratio"] = (
                        float(own_norm_ratio)
                    )
                    metrics[
                        f"{safe_domain}/token_sequence_closure/"
                        "all_tokens_vs_domain_norm_ratio_error"
                    ] = abs(float(own_norm_ratio) - 1.0)
        if availability:
            available_count = sum(value > 0.5 for value in availability)
            metrics[f"{safe_domain}/token_grad_cost/available_token_count"] = float(available_count)
            metrics[f"{safe_domain}/token_grad_cost/unavailable_token_count"] = float(
                len(availability) - available_count
            )
            metrics[f"{safe_domain}/token_grad_cost/valid_frac"] = available_count / len(availability)
        if seconds:
            seconds_sum = sum(seconds)
            metrics[f"{safe_domain}/token_grad_cost/seconds_sum"] = seconds_sum
            metrics[f"{safe_domain}/token_grad_cost/seconds_mean"] = _mean(seconds) or 0.0
            metrics[f"{safe_domain}/token_grad_cost/seconds_per_selected_token"] = seconds_sum / max(
                selected_token_total,
                1.0,
            )
        if autograd_seconds:
            metrics[f"{safe_domain}/token_grad_cost/autograd_seconds_sum"] = sum(autograd_seconds)
        if fallback_seconds:
            metrics[f"{safe_domain}/token_grad_cost/backward_fallback_seconds_sum"] = sum(fallback_seconds)
        if fallback_used:
            metrics[f"{safe_domain}/token_grad_cost/backward_fallback_count"] = float(
                sum(value > 0.5 for value in fallback_used)
            )
        if pre_restore_rel_l2:
            metrics[f"{safe_domain}/token_grad_cost/restore_pre_target_rel_l2_max"] = max(pre_restore_rel_l2)
        if post_restore_rel_l2:
            metrics[f"{safe_domain}/token_grad_cost/restore_post_target_rel_l2_max"] = max(post_restore_rel_l2)
        if pre_restore_max_abs:
            metrics[f"{safe_domain}/token_grad_cost/restore_pre_target_max_abs_max"] = max(pre_restore_max_abs)
        if post_restore_max_abs:
            metrics[f"{safe_domain}/token_grad_cost/restore_post_target_max_abs_max"] = max(post_restore_max_abs)
        if original_restore_rel_l2:
            metrics[f"{safe_domain}/token_grad_cost/restore_original_rel_l2_max"] = max(original_restore_rel_l2)
        if original_restore_max_abs:
            metrics[f"{safe_domain}/token_grad_cost/restore_original_max_abs_max"] = max(original_restore_max_abs)
        if finite_norms:
            metrics[f"{safe_domain}/token_grad/norm_mean"] = _mean(finite_norms) or 0.0
            metrics[f"{safe_domain}/token_grad/norm_p95"] = _percentile(finite_norms, 95.0) or 0.0
            metrics[f"{safe_domain}/token_grad/norm_max"] = max(finite_norms)
        if other_cos:
            conflicts = [max(0.0, -value) for value in other_cos]
            metrics[f"{safe_domain}/token_grad_conflict/other_cos_mean"] = _mean(other_cos) or 0.0
            metrics[f"{safe_domain}/token_grad_conflict/other_cos_p05"] = _percentile(other_cos, 5.0) or 0.0
            metrics[f"{safe_domain}/token_grad_conflict/other_cos_negative_frac"] = float(
                np.mean([value < 0.0 for value in other_cos])
            )
            metrics[f"{safe_domain}/token_grad_conflict/conflict_to_other_mean"] = _mean(conflicts) or 0.0
            metrics[f"{safe_domain}/token_grad_conflict/conflict_to_other_max"] = max(conflicts)
        if own_projection:
            metrics[f"{safe_domain}/token_grad_contribution/own_projection_share_mean"] = (
                _mean(own_projection) or 0.0
            )
            metrics[f"{safe_domain}/token_grad_contribution/own_projection_share_sum"] = sum(own_projection)
        if other_projection:
            metrics[f"{safe_domain}/token_grad_contribution/other_projection_share_mean"] = (
                _mean(other_projection) or 0.0
            )
            metrics[f"{safe_domain}/token_grad_contribution/negative_other_projection_share_sum"] = sum(
                max(0.0, -value) for value in other_projection
            )
        return metrics

    def _summarize_global_token_gradient_cost(
        self,
        rows: list[dict[str, Any]],
        total_seconds: float,
    ) -> dict[str, float]:
        selected_token_total = sum(
            float(row.get("selected_token_count", 1.0) or 0.0) for row in rows
        )
        metrics: dict[str, float] = {
            "global/token_grad_cost/seconds": float(total_seconds),
            "global/token_grad_cost/selected_token_count": float(selected_token_total),
            "global/token_grad_cost/max_memory_allocated_gb": _max_memory_allocated_gb(),
        }
        candidate_token_counts: dict[str, float] = {}
        candidate_sample_counts: dict[str, float] = {}
        candidate_gap_mass: dict[str, float] = {}
        candidate_gap_abs_mass: dict[str, float] = {}
        candidate_loss_abs_mass: dict[str, float] = {}
        for row in rows:
            domain = str(row.get("domain", "unknown"))
            candidate_token_counts[domain] = max(
                candidate_token_counts.get(domain, 0.0),
                float(row.get("global_candidate_token_count", 0.0) or 0.0),
            )
            candidate_sample_counts[domain] = max(
                candidate_sample_counts.get(domain, 0.0),
                float(row.get("global_candidate_sample_count", 0.0) or 0.0),
            )
            candidate_gap_mass[domain] = max(
                candidate_gap_mass.get(domain, 0.0),
                float(row.get("global_candidate_gap_mass", 0.0) or 0.0),
            )
            candidate_gap_abs_mass[domain] = max(
                candidate_gap_abs_mass.get(domain, 0.0),
                float(row.get("global_candidate_gap_abs_mass", 0.0) or 0.0),
            )
            candidate_loss_abs_mass[domain] = max(
                candidate_loss_abs_mass.get(domain, 0.0),
                float(row.get("global_candidate_loss_abs_mass", 0.0) or 0.0),
            )
        if candidate_token_counts:
            metrics["global/token_grad_cost/global_candidate_token_count"] = float(
                sum(candidate_token_counts.values())
            )
            metrics["global/token_grad_cost/global_candidate_sample_count"] = float(
                sum(candidate_sample_counts.values())
            )
            metrics["global/token_grad_cost/global_candidate_gap_mass"] = float(
                sum(candidate_gap_mass.values())
            )
            metrics["global/token_grad_cost/global_candidate_gap_abs_mass"] = float(
                sum(candidate_gap_abs_mass.values())
            )
            metrics["global/token_grad_cost/global_candidate_loss_abs_mass"] = float(
                sum(candidate_loss_abs_mass.values())
            )
        metrics["global/token_grad_cost/selected_sample_count"] = float(
            sum(float(row.get("selected_sample_count", 0.0) or 0.0) for row in rows)
        )
        if selected_token_total > 0.0:
            metrics["global/token_grad_cost/seconds_per_selected_token"] = float(total_seconds) / selected_token_total

        availability = _finite_values([row.get("token_grad_available") for row in rows])
        if availability:
            available_count = sum(value > 0.5 for value in availability)
            metrics["global/token_grad_cost/available_token_count"] = float(available_count)
            metrics["global/token_grad_cost/unavailable_token_count"] = float(
                len(availability) - available_count
            )
            metrics["global/token_grad_cost/valid_frac"] = available_count / len(availability)

        autograd_seconds = _finite_values([row.get("token_grad_autograd_seconds") for row in rows])
        if autograd_seconds:
            metrics["global/token_grad_cost/autograd_seconds_sum"] = sum(autograd_seconds)

        fallback_seconds = _finite_values([row.get("token_grad_backward_fallback_seconds") for row in rows])
        if fallback_seconds:
            metrics["global/token_grad_cost/backward_fallback_seconds_sum"] = sum(fallback_seconds)

        fallback_used = _finite_values([row.get("token_grad_backward_fallback_used") for row in rows])
        if fallback_used:
            metrics["global/token_grad_cost/backward_fallback_count"] = float(
                sum(value > 0.5 for value in fallback_used)
            )
        pre_restore_rel_l2 = _finite_values([row.get("token_grad_restore_pre_target_rel_l2") for row in rows])
        if pre_restore_rel_l2:
            metrics["global/token_grad_cost/restore_pre_target_rel_l2_max"] = max(pre_restore_rel_l2)
        post_restore_rel_l2 = _finite_values([row.get("token_grad_restore_post_target_rel_l2") for row in rows])
        if post_restore_rel_l2:
            metrics["global/token_grad_cost/restore_post_target_rel_l2_max"] = max(post_restore_rel_l2)
        pre_restore_max_abs = _finite_values([row.get("token_grad_restore_pre_target_max_abs") for row in rows])
        if pre_restore_max_abs:
            metrics["global/token_grad_cost/restore_pre_target_max_abs_max"] = max(pre_restore_max_abs)
        post_restore_max_abs = _finite_values([row.get("token_grad_restore_post_target_max_abs") for row in rows])
        if post_restore_max_abs:
            metrics["global/token_grad_cost/restore_post_target_max_abs_max"] = max(post_restore_max_abs)
        original_restore_rel_l2 = _finite_values([row.get("token_grad_restore_original_rel_l2") for row in rows])
        if original_restore_rel_l2:
            metrics["global/token_grad_cost/restore_original_rel_l2_max"] = max(original_restore_rel_l2)
        original_restore_max_abs = _finite_values([row.get("token_grad_restore_original_max_abs") for row in rows])
        if original_restore_max_abs:
            metrics["global/token_grad_cost/restore_original_max_abs_max"] = max(original_restore_max_abs)
        return metrics

    def _recompute_sample_to_domain_stats(
        self,
        micro_batch: DataProto,
        *,
        target_chunks: tuple[torch.Tensor, ...],
        target_norm: float,
        target_norm_sq: float,
        restore_target_map: dict[str, tuple[tuple[torch.Tensor, ...], float]] | None = None,
        loss_scale_factor: float,
        on_policy: bool,
    ) -> dict[str, float | None]:
        started_at = time.perf_counter()
        parameters = _trainable_parameters(self.actor)
        if len(parameters) != len(target_chunks):
            return {
                "sample_to_domain_cos": None,
                "sample_projection_share": None,
                "sample_projection_share_normalized": None,
                "sample_recompute_grad_norm": None,
                "sample_recompute_non_none_grad_count": 0.0,
                "sample_recompute_available": 0.0,
                "sample_recompute_autograd_error": "parameter_target_mismatch",
                "sample_recompute_backward_used": float(self.sample_gradient_backward_recompute_enabled),
                "sample_recompute_backward_sync_used": float(self.sample_gradient_backward_sync_enabled),
                "sample_recompute_seconds": time.perf_counter() - started_at,
            }
        actor_config = getattr(self.actor, "config", {})
        loss_agg_mode = str(_cfg_get(actor_config, "loss_agg_mode", "token-mean"))
        replica_count = (
            _gradient_replica_count(self.actor)
            if self._sample_gradient_uses_full_local_params
            else 1
        )
        token_mask = None
        effective_loss_scale_factor = float(loss_scale_factor)
        use_backward_recompute = bool(self.sample_gradient_backward_recompute_enabled)
        grad_dtypes = _parameter_grad_dtypes(parameters)
        grad_snapshot = (
            _snapshot_parameter_grads_for_restore(parameters)
            if use_backward_recompute and restore_target_map is None
            else None
        )
        pre_diff = (
            _parameter_grad_target_diff_stats(parameters, restore_target_map)
            if use_backward_recompute and restore_target_map is not None
            else {}
        )
        post_diff: dict[str, float] = {}
        try:
            response_mask = micro_batch.batch["response_mask"]
        except Exception:
            response_mask = None
        if response_mask is not None:
            token_mask = response_mask.detach().float().clone()
            contribution_scale = _token_mask_contribution_scale(
                response_mask,
                token_mask,
                loss_agg_mode,
            )
            if contribution_scale <= 0.0:
                return {
                    "sample_to_domain_cos": None,
                    "sample_projection_share": None,
                    "sample_projection_share_normalized": None,
                    "sample_recompute_grad_norm": None,
                    "sample_recompute_non_none_grad_count": 0.0,
                    "sample_recompute_available": 0.0,
                    "sample_recompute_autograd_error": "zero_contribution_scale",
                    "sample_recompute_backward_used": float(use_backward_recompute),
                    "sample_recompute_backward_sync_used": float(self.sample_gradient_backward_sync_enabled),
                    "sample_recompute_seconds": time.perf_counter() - started_at,
                }
            effective_loss_scale_factor *= contribution_scale
        gradients: tuple[torch.Tensor | None, ...] = tuple(None for _ in parameters)
        autograd_error: str | None = None
        try:
            if use_backward_recompute:
                _clear_parameter_grads(parameters)
            recompute_sync_context = (
                nullcontext()
                if not use_backward_recompute or self.sample_gradient_backward_sync_enabled
                else _actor_no_sync_context(self.actor)
            )
            with recompute_sync_context:
                loss = _actor_micro_batch_loss(
                    self.actor,
                    micro_batch,
                    loss_scale_factor=effective_loss_scale_factor,
                    on_policy=on_policy,
                    safe_logprob_backward=not use_backward_recompute,
                    response_mask_override=token_mask,
                )
                if use_backward_recompute:
                    loss.backward()
                    _finalize_fsdp_after_auxiliary_backward(self.actor)
                    gradients = tuple(parameter.grad for parameter in parameters)
                else:
                    gradients = torch.autograd.grad(loss, parameters, retain_graph=False, allow_unused=True)
            local_norm_sq, local_dot = self._grad_stats_from_tensors(gradients, target_chunks)
            non_none_grad_count = sum(gradient is not None for gradient in gradients)
        except Exception as exc:
            local_norm_sq, local_dot = None, None
            non_none_grad_count = 0
            autograd_error = type(exc).__name__
        finally:
            _finalize_fsdp_after_auxiliary_backward(self.actor)
            if use_backward_recompute:
                if restore_target_map is not None:
                    _restore_parameter_grads_from_targets(parameters, restore_target_map, grad_dtypes=grad_dtypes)
                    post_diff = _parameter_grad_target_diff_stats(parameters, restore_target_map)
                elif grad_snapshot is not None:
                    _restore_parameter_grads_from_snapshot(parameters, grad_snapshot, grad_dtypes=grad_dtypes)
        if non_none_grad_count == 0 and autograd_error is None:
            autograd_error = "all_parameters_disconnected"
        restore_stats = {
            "sample_recompute_restore_pre_target_rel_l2": pre_diff.get("rel_l2", 0.0),
            "sample_recompute_restore_pre_target_max_abs": pre_diff.get("max_abs", 0.0),
            "sample_recompute_restore_post_target_rel_l2": post_diff.get("rel_l2", 0.0),
            "sample_recompute_restore_post_target_max_abs": post_diff.get("max_abs", 0.0),
            "sample_recompute_restore_target_norm": pre_diff.get("target_norm", 0.0),
        }
        if local_norm_sq is None or local_dot is None:
            return {
                "sample_to_domain_cos": None,
                "sample_projection_share": None,
                "sample_projection_share_normalized": None,
                "sample_recompute_grad_norm": None,
                "sample_recompute_non_none_grad_count": float(non_none_grad_count),
                "sample_recompute_available": 0.0,
                "sample_recompute_autograd_error": autograd_error,
                "sample_recompute_backward_used": float(use_backward_recompute),
                "sample_recompute_backward_sync_used": float(self.sample_gradient_backward_sync_enabled),
                "sample_recompute_seconds": time.perf_counter() - started_at,
                **restore_stats,
            }
        if self._sample_gradient_uses_full_local_params:
            norm_sq = max(local_norm_sq, 0.0)
            dot = local_dot
        else:
            norm_sq = max(_all_reduce_sum(local_norm_sq), 0.0)
            dot = _all_reduce_sum(local_dot)
        if non_none_grad_count > 0 and norm_sq <= 0.0:
            self._sample_zero_norm_count += 1
        sample_norm_raw = norm_sq**0.5
        local_to_global_scale = (
            1.0 / max(replica_count, 1)
            if use_backward_recompute and not self.sample_gradient_backward_sync_enabled
            else 1.0
        )
        target_norm_for_cosine = target_norm
        sample_norm = sample_norm_raw * abs(local_to_global_scale)
        available = non_none_grad_count > 0 and sample_norm_raw > 0.0 and target_norm_for_cosine > 0.0
        cosine = dot / (sample_norm_raw * target_norm_for_cosine) if available else None
        projection_share_raw = (
            dot / target_norm_sq
            if available and target_norm_sq > 0.0
            else None
        )
        projection_share = (
            projection_share_raw * local_to_global_scale
            if projection_share_raw is not None
            else None
        )
        return {
            "sample_to_domain_cos": cosine,
            "sample_projection_share": projection_share,
            "sample_projection_share_normalized": None,
            "sample_projection_share_raw": projection_share_raw,
            "sample_projection_share_scale": local_to_global_scale,
            "sample_recompute_grad_norm": sample_norm,
            "sample_recompute_grad_norm_raw": sample_norm_raw,
            "sample_recompute_non_none_grad_count": float(non_none_grad_count),
            "sample_recompute_available": float(available),
            "sample_recompute_autograd_error": autograd_error,
            "sample_recompute_backward_used": float(use_backward_recompute),
            "sample_recompute_backward_sync_used": float(self.sample_gradient_backward_sync_enabled),
            "sample_recompute_replica_count": float(replica_count),
            "sample_recompute_seconds": time.perf_counter() - started_at,
            **restore_stats,
        }

    def _grad_stats_from_tensors(
        self,
        gradients: tuple[torch.Tensor | None, ...],
        target_chunks: tuple[torch.Tensor, ...],
    ) -> tuple[float | None, float | None]:
        local_norm_sq = 0.0
        local_dot = 0.0
        for gradient, target in zip(gradients, target_chunks):
            if gradient is None:
                continue
            gradient_gpu = gradient.detach().reshape(-1).float()
            if gradient_gpu.numel() != target.numel():
                return None, None
            target_gpu = target.reshape(-1).to(device=gradient_gpu.device, dtype=torch.float32)
            gradient_norm_sq = _chunked_vector_dot(gradient_gpu, gradient_gpu)
            gradient_target_dot = _chunked_vector_dot(gradient_gpu, target_gpu)
            if gradient_norm_sq is None or gradient_target_dot is None:
                return None, None
            local_norm_sq += gradient_norm_sq
            local_dot += gradient_target_dot
        return local_norm_sq, local_dot
