"""Top-k distillation helpers shared by verl workers and tests."""

from __future__ import annotations

from typing import Any

import torch


_TOPK_LOGPROB_CHUNK_SIZE = 16

CHOSEN_TOKEN_REVERSE_KL = "chosen_token_reverse_kl"
CHOSEN_TOKEN_POLICY_GRADIENT = "chosen_token_policy_gradient"
DISTILL_LOSS_BUILDER_AUTO = "auto"
DISTILL_LOSS_BUILDER_CHOSEN_TOKEN_REVERSE_KL = "chosen_token_reverse_kl"
DISTILL_LOSS_BUILDER_POLICY_GRADIENT = "policy_gradient"
DISTILL_LOSS_BUILDER_TOPK_KL = "topk_kl"
DISTILL_LOSS_BUILDERS = {
    DISTILL_LOSS_BUILDER_AUTO,
    DISTILL_LOSS_BUILDER_CHOSEN_TOKEN_REVERSE_KL,
    DISTILL_LOSS_BUILDER_POLICY_GRADIENT,
    DISTILL_LOSS_BUILDER_TOPK_KL,
}
DISTILL_LOSS_BUILDER_ALIASES = {
    "pg": DISTILL_LOSS_BUILDER_POLICY_GRADIENT,
    "chosen_token_pg": DISTILL_LOSS_BUILDER_POLICY_GRADIENT,
    CHOSEN_TOKEN_POLICY_GRADIENT: DISTILL_LOSS_BUILDER_POLICY_GRADIENT,
    "topk": DISTILL_LOSS_BUILDER_TOPK_KL,
    "topk_distill": DISTILL_LOSS_BUILDER_TOPK_KL,
    "topk_distillation": DISTILL_LOSS_BUILDER_TOPK_KL,
}
TOPK_LOGPROB_MODE_SPARSE = "sparse"
TOPK_LOGPROB_MODE_FULL_VOCAB = "full_vocab"
TOPK_LOGPROB_MODES = {TOPK_LOGPROB_MODE_SPARSE, TOPK_LOGPROB_MODE_FULL_VOCAB}
TOPK_SUPPORT_SOURCE_TEACHER = "teacher"
TOPK_SUPPORT_SOURCE_STUDENT = "student"
TOPK_SUPPORT_SOURCES = {TOPK_SUPPORT_SOURCE_TEACHER, TOPK_SUPPORT_SOURCE_STUDENT}
TOPK_FORWARD_KL_WITH_TAIL = "topk_forward_kl_with_tail"
TOPK_REVERSE_KL_WITH_TAIL = "topk_reverse_kl_with_tail"
TOPK_RENORMALIZED_FORWARD_KL = "topk_renormalized_forward_kl"
TOPK_RENORMALIZED_REVERSE_KL = "topk_renormalized_reverse_kl"
NAIVE_RENORMALIZED_TOPK_KL = "naive_renormalized_topk_kl"

TOPK_RENORMALIZED_MODES = {
    TOPK_RENORMALIZED_FORWARD_KL,
    TOPK_RENORMALIZED_REVERSE_KL,
    NAIVE_RENORMALIZED_TOPK_KL,
}

TOPK_DISTILL_MODES = {
    TOPK_FORWARD_KL_WITH_TAIL,
    TOPK_REVERSE_KL_WITH_TAIL,
    TOPK_RENORMALIZED_FORWARD_KL,
    TOPK_RENORMALIZED_REVERSE_KL,
    NAIVE_RENORMALIZED_TOPK_KL,
}

TOPK_REVERSE_KL_MODES = {
    TOPK_REVERSE_KL_WITH_TAIL,
    TOPK_RENORMALIZED_REVERSE_KL,
    NAIVE_RENORMALIZED_TOPK_KL,
}

TEACHER_PREFIX_PREFIX_AND_SUFFIX = "prefix_and_suffix"
TEACHER_PREFIX_SUFFIX_ONLY = "suffix_only"
TEACHER_PREFIX_PREFIX_ONLY = "prefix_only"
TEACHER_PREFIX_LOSS_REGIONS = {
    TEACHER_PREFIX_PREFIX_AND_SUFFIX,
    TEACHER_PREFIX_SUFFIX_ONLY,
    TEACHER_PREFIX_PREFIX_ONLY,
}


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def distill_mode(policy_loss_config: Any) -> str:
    return str(cfg_get(policy_loss_config, "distill_mode", CHOSEN_TOKEN_REVERSE_KL))


def distill_loss_builder(policy_loss_config: Any) -> str:
    builder = str(
        cfg_get(policy_loss_config, "distill_loss_builder", DISTILL_LOSS_BUILDER_AUTO)
        or DISTILL_LOSS_BUILDER_AUTO
    ).lower()
    builder = DISTILL_LOSS_BUILDER_ALIASES.get(builder, builder)
    if builder != DISTILL_LOSS_BUILDER_AUTO:
        if builder not in DISTILL_LOSS_BUILDERS:
            raise ValueError(
                "distill_loss_builder must be one of "
                f"{sorted(DISTILL_LOSS_BUILDERS)} or aliases "
                f"{sorted(DISTILL_LOSS_BUILDER_ALIASES)}, got {builder!r}."
            )
        return builder

    mode = distill_mode(policy_loss_config)
    if mode in {CHOSEN_TOKEN_POLICY_GRADIENT, DISTILL_LOSS_BUILDER_POLICY_GRADIENT}:
        return DISTILL_LOSS_BUILDER_POLICY_GRADIENT
    if mode in TOPK_DISTILL_MODES or bool(cfg_get(policy_loss_config, "topk_distill_enabled", False)):
        return DISTILL_LOSS_BUILDER_TOPK_KL
    return DISTILL_LOSS_BUILDER_CHOSEN_TOKEN_REVERSE_KL


def uses_topk_distill_loss(policy_loss_config: Any) -> bool:
    return distill_loss_builder(policy_loss_config) == DISTILL_LOSS_BUILDER_TOPK_KL


def resolved_topk_distill_mode(policy_loss_config: Any) -> str:
    mode = distill_mode(policy_loss_config)
    if mode == CHOSEN_TOKEN_REVERSE_KL and bool(cfg_get(policy_loss_config, "topk_distill_enabled", False)):
        direction = str(cfg_get(policy_loss_config, "topk_distill_kl_direction", "reverse")).lower()
        if direction == "forward":
            return TOPK_RENORMALIZED_FORWARD_KL
        return TOPK_RENORMALIZED_REVERSE_KL
    return mode


def is_topk_distill_enabled(policy_loss_config: Any) -> bool:
    mode = distill_mode(policy_loss_config)
    return bool(cfg_get(policy_loss_config, "topk_distill_enabled", False)) or mode in TOPK_DISTILL_MODES


def topk_distill_k(policy_loss_config: Any) -> int:
    return max(1, int(cfg_get(policy_loss_config, "topk_distill_k", 8) or 8))


def topk_distill_support_source(policy_loss_config: Any) -> str:
    source = str(
        cfg_get(policy_loss_config, "topk_distill_support_source", TOPK_SUPPORT_SOURCE_TEACHER)
    ).lower()
    if source not in TOPK_SUPPORT_SOURCES:
        raise ValueError(
            "topk_distill_support_source must be one of "
            f"{sorted(TOPK_SUPPORT_SOURCES)}, got {source!r}."
        )
    return source


def topk_distill_weight(policy_loss_config: Any) -> float:
    return float(cfg_get(policy_loss_config, "topk_distill_loss_weight", 1.0) or 0.0)


def topk_distill_temperature(policy_loss_config: Any) -> float:
    value = float(cfg_get(policy_loss_config, "topk_distill_temperature", 1.0) or 1.0)
    return max(value, 1e-6)


def topk_distill_logprob_chunk_size(policy_loss_config: Any) -> int:
    return max(1, int(cfg_get(policy_loss_config, "topk_distill_logprob_chunk_size", _TOPK_LOGPROB_CHUNK_SIZE) or 1))


def topk_distill_logprob_mode(policy_loss_config: Any) -> str:
    mode = str(cfg_get(policy_loss_config, "topk_distill_logprob_mode", TOPK_LOGPROB_MODE_SPARSE)).lower()
    if mode not in TOPK_LOGPROB_MODES:
        raise ValueError(f"topk_distill_logprob_mode must be one of {sorted(TOPK_LOGPROB_MODES)}, got {mode!r}.")
    return mode


def topk_distill_include_tail(policy_loss_config: Any) -> bool:
    mode = resolved_topk_distill_mode(policy_loss_config)
    if mode in TOPK_RENORMALIZED_MODES:
        return False
    return bool(cfg_get(policy_loss_config, "topk_distill_tail_bucket", True))


def topk_distill_uses_renormalized_support(policy_loss_config: Any) -> bool:
    return resolved_topk_distill_mode(policy_loss_config) in TOPK_RENORMALIZED_MODES


def is_teacher_prefix_enabled(policy_loss_config: Any) -> bool:
    return bool(cfg_get(policy_loss_config, "teacher_prefix_enabled", False))


def teacher_prefix_loss_region(policy_loss_config: Any) -> str:
    region = str(
        cfg_get(policy_loss_config, "teacher_prefix_loss_region", TEACHER_PREFIX_SUFFIX_ONLY)
    ).lower()
    if region == "all":
        return TEACHER_PREFIX_PREFIX_AND_SUFFIX
    if region not in TEACHER_PREFIX_LOSS_REGIONS:
        raise ValueError(
            "teacher_prefix_loss_region must be one of "
            f"{sorted(TEACHER_PREFIX_LOSS_REGIONS)} or 'all', got {region!r}."
        )
    return region


def teacher_prefix_forward_weight(policy_loss_config: Any) -> float:
    return float(cfg_get(policy_loss_config, "teacher_prefix_forward_kl_weight", 1.0) or 0.0)


def teacher_type_at(opd_teacher: object, index: int) -> object:
    if hasattr(opd_teacher, "ndim"):
        if opd_teacher.ndim == 0:
            return opd_teacher.item()
        return opd_teacher[index]
    if isinstance(opd_teacher, (list, tuple)):
        return opd_teacher[index]
    return opd_teacher


def select_teacher_log_prob_tensor(
    model_inputs: dict[str, Any],
    policy_loss_config: Any,
    *,
    math_key: str = "math_teacher_log_prob",
    code_key: str = "code_teacher_log_prob",
) -> torch.Tensor:
    if math_key not in model_inputs:
        raise ValueError(f"Teacher log-prob selection requires {math_key!r} in model_inputs.")
    math_log_prob = model_inputs[math_key]
    code_log_prob = model_inputs.get(code_key, math_log_prob)
    if not bool(cfg_get(policy_loss_config, "multi_teacher_distill", False)) or "opd_teacher" not in model_inputs:
        return math_log_prob

    opd_teacher = model_inputs["opd_teacher"]
    selected = torch.empty_like(math_log_prob)
    for idx in range(int(math_log_prob.shape[0])):
        teacher_type = teacher_type_at(opd_teacher, idx)
        if teacher_type == "code" and code_key in model_inputs:
            selected[idx] = code_log_prob[idx]
        else:
            selected[idx] = math_log_prob[idx]
    return selected


def teacher_prefix_masks(
    model_inputs: dict[str, Any],
    response_mask: torch.Tensor,
    policy_loss_config: Any,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return prefix-forward-KL and suffix-distillation masks.

    The prefix mask is expected to mark tokens sampled from the teacher. Suffix
    tokens remain student-sampled and keep the existing OPD/top-k objective.
    """

    if not is_teacher_prefix_enabled(policy_loss_config) or "teacher_prefix_mask" not in model_inputs:
        empty = torch.zeros_like(response_mask)
        return empty, response_mask, False

    prefix_mask = model_inputs["teacher_prefix_mask"].to(device=response_mask.device, dtype=response_mask.dtype)
    prefix_mask = prefix_mask * response_mask
    suffix_mask = (response_mask - prefix_mask).clamp(min=0.0, max=1.0)
    region = teacher_prefix_loss_region(policy_loss_config)
    if region == TEACHER_PREFIX_SUFFIX_ONLY:
        prefix_loss_mask = torch.zeros_like(response_mask)
    else:
        prefix_loss_mask = prefix_mask
    if "student_suffix_mask" in model_inputs:
        suffix_mask = model_inputs["student_suffix_mask"].to(device=response_mask.device, dtype=response_mask.dtype)
        suffix_mask = suffix_mask * response_mask
    if region == TEACHER_PREFIX_PREFIX_ONLY:
        suffix_mask = torch.zeros_like(response_mask)
    active = bool(prefix_mask.detach().sum().item() > 0.0)
    return prefix_loss_mask, suffix_mask, active


def chosen_token_forward_kl_matrix(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
) -> torch.Tensor:
    return teacher_log_probs.float() - student_log_probs.float()


def chosen_token_policy_gradient_reward_matrix(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
) -> torch.Tensor:
    return teacher_log_probs.float() - student_log_probs.float()


class _SelectedLogitsFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        hidden_states: torch.Tensor,
        vocab_weights: torch.Tensor,
        token_ids: torch.Tensor,
        bias: torch.Tensor | None,
        temperature: float,
        chunk_size: int,
    ) -> torch.Tensor:
        if hidden_states.dim() < 2:
            raise ValueError(f"hidden_states must have at least 2 dims, got {tuple(hidden_states.shape)}")
        if token_ids.dim() != hidden_states.dim():
            raise ValueError(
                "token_ids must have one more support dimension in place of hidden_states hidden dimension, "
                f"got hidden_states={tuple(hidden_states.shape)} token_ids={tuple(token_ids.shape)}."
            )
        if tuple(hidden_states.shape[:-1]) != tuple(token_ids.shape[:-1]):
            raise ValueError(
                "token_ids prefix shape must match hidden_states prefix shape, "
                f"got hidden_states={tuple(hidden_states.shape)} token_ids={tuple(token_ids.shape)}."
            )
        if vocab_weights.dim() != 2:
            raise ValueError(f"vocab_weights must be 2-D, got {tuple(vocab_weights.shape)}")
        if int(vocab_weights.shape[-1]) != int(hidden_states.shape[-1]):
            raise ValueError(
                "vocab_weights hidden dimension must match hidden_states, "
                f"got {tuple(vocab_weights.shape)} and {tuple(hidden_states.shape)}."
            )
        if bias is not None and int(bias.shape[0]) != int(vocab_weights.shape[0]):
            raise ValueError(
                f"bias vocab dimension must match vocab_weights, got {tuple(bias.shape)} and {tuple(vocab_weights.shape)}."
            )
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        ctx.set_materialize_grads(False)
        token_ids = token_ids.to(device=hidden_states.device, dtype=torch.long)
        hidden_shape = tuple(hidden_states.shape)
        support_size = int(token_ids.shape[-1])
        hidden_size = int(hidden_states.shape[-1])
        flat_hidden = hidden_states.reshape(-1, hidden_size)
        flat_ids = token_ids.reshape(-1, support_size)
        output = hidden_states.new_empty(flat_ids.shape)
        inv_temperature = 1.0 / max(float(temperature), 1e-6)

        for start in range(0, int(flat_hidden.shape[0]), int(chunk_size)):
            end = min(start + int(chunk_size), int(flat_hidden.shape[0]))
            ids_chunk = flat_ids[start:end]
            weight_chunk = vocab_weights.index_select(0, ids_chunk.reshape(-1)).view(
                end - start,
                support_size,
                hidden_size,
            )
            logits_chunk = torch.bmm(weight_chunk, flat_hidden[start:end].unsqueeze(-1)).squeeze(-1)
            if bias is not None:
                logits_chunk = logits_chunk + bias.index_select(0, ids_chunk.reshape(-1)).view(end - start, support_size)
            output[start:end] = logits_chunk * inv_temperature

        ctx.save_for_backward(hidden_states, vocab_weights, token_ids)
        ctx.has_bias = bias is not None
        ctx.bias_requires_grad = bool(bias is not None and bias.requires_grad)
        ctx.temperature = max(float(temperature), 1e-6)
        ctx.chunk_size = int(chunk_size)
        ctx.hidden_shape = hidden_shape
        ctx.support_size = support_size
        return output.reshape(*hidden_shape[:-1], support_size)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, None, torch.Tensor | None, None, None]:
        if grad_output is None:
            return None, None, None, None, None, None

        hidden_states, vocab_weights, token_ids = ctx.saved_tensors
        support_size = int(ctx.support_size)
        hidden_size = int(hidden_states.shape[-1])
        flat_hidden = hidden_states.reshape(-1, hidden_size)
        flat_ids = token_ids.reshape(-1, support_size)
        flat_grad = grad_output.reshape(-1, support_size).to(dtype=vocab_weights.dtype)
        inv_temperature = 1.0 / max(float(ctx.temperature), 1e-6)

        grad_hidden = torch.zeros_like(flat_hidden) if hidden_states.requires_grad else None
        grad_weights = torch.zeros_like(vocab_weights) if vocab_weights.requires_grad else None
        grad_bias = (
            torch.zeros(vocab_weights.shape[0], device=vocab_weights.device, dtype=vocab_weights.dtype)
            if ctx.bias_requires_grad
            else None
        )

        for start in range(0, int(flat_hidden.shape[0]), int(ctx.chunk_size)):
            end = min(start + int(ctx.chunk_size), int(flat_hidden.shape[0]))
            ids_chunk = flat_ids[start:end]
            flat_ids_chunk = ids_chunk.reshape(-1)
            grad_chunk = flat_grad[start:end] * inv_temperature

            if grad_hidden is not None:
                weight_chunk = vocab_weights.index_select(0, flat_ids_chunk).view(
                    end - start,
                    support_size,
                    hidden_size,
                )
                grad_hidden[start:end] = torch.bmm(
                    weight_chunk.transpose(1, 2),
                    grad_chunk.unsqueeze(-1),
                ).squeeze(-1)

            if grad_weights is not None:
                weight_grad_chunk = (
                    grad_chunk.unsqueeze(-1) * flat_hidden[start:end].to(dtype=grad_chunk.dtype).unsqueeze(1)
                ).reshape(-1, hidden_size)
                grad_weights.index_add_(0, flat_ids_chunk, weight_grad_chunk)

            if grad_bias is not None:
                grad_bias.index_add_(0, flat_ids_chunk, grad_chunk.reshape(-1))

        if grad_hidden is not None:
            grad_hidden = grad_hidden.reshape(ctx.hidden_shape)
        return grad_hidden, grad_weights, None, grad_bias, None, None


def selected_logits_from_hidden_states(
    hidden_states: torch.Tensor,
    *,
    vocab_weights: torch.Tensor,
    token_ids: torch.Tensor,
    bias: torch.Tensor | None = None,
    temperature: float = 1.0,
    chunk_size: int = _TOPK_LOGPROB_CHUNK_SIZE,
) -> torch.Tensor:
    """Compute logits only for a per-position support set.

    This avoids materializing ``[tokens, vocab]`` logits for renormalized
    top-k distillation.  The backward pass scatters gradients directly into
    the dense LM-head parameter gradient without creating a dense logits
    gradient buffer.
    """

    return _SelectedLogitsFunction.apply(
        hidden_states,
        vocab_weights,
        token_ids,
        bias,
        float(temperature),
        max(1, int(chunk_size)),
    )


def topk_log_probs_from_logits(
    logits: torch.Tensor,
    *,
    topk: int | None = None,
    gather_topk_ids: torch.Tensor | None = None,
    normalize_gathered: bool = True,
    chunk_size: int = _TOPK_LOGPROB_CHUNK_SIZE,
    logprob_mode: str = TOPK_LOGPROB_MODE_SPARSE,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Compute top-k log-probs with sparse or full-vocab normalization."""

    if topk is None and gather_topk_ids is None:
        return None, None, None
    if logits.dim() < 2:
        raise ValueError(f"logits must have at least 2 dims, got shape {tuple(logits.shape)}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    logprob_mode = str(logprob_mode).lower()
    if logprob_mode not in TOPK_LOGPROB_MODES:
        raise ValueError(f"logprob_mode must be one of {sorted(TOPK_LOGPROB_MODES)}, got {logprob_mode!r}.")

    vocab_size = int(logits.shape[-1])
    prefix_shape = tuple(logits.shape[:-1])
    flat_logits = logits.reshape(-1, vocab_size)
    topk_count = min(max(1, int(topk)), vocab_size) if topk is not None else None

    flat_gather_ids = None
    gather_shape = None
    if gather_topk_ids is not None:
        gather_ids = gather_topk_ids.to(device=logits.device, dtype=torch.long)
        if tuple(gather_ids.shape[:-1]) != prefix_shape:
            raise ValueError(
                "gather_topk_ids must match logits prefix shape, "
                f"got {tuple(gather_ids.shape)} for logits {tuple(logits.shape)}."
            )
        gather_shape = tuple(gather_ids.shape[-1:])
        flat_gather_ids = gather_ids.reshape(-1, int(gather_ids.shape[-1]))

    topk_id_chunks: list[torch.Tensor] = []
    topk_log_prob_chunks: list[torch.Tensor] = []
    gathered_log_prob_chunks: list[torch.Tensor] = []
    for start in range(0, int(flat_logits.shape[0]), chunk_size):
        end = min(start + chunk_size, int(flat_logits.shape[0]))
        needs_vocab_normalizer = (
            topk_count is not None
            or logprob_mode == TOPK_LOGPROB_MODE_FULL_VOCAB
            or (flat_gather_ids is not None and normalize_gathered)
        )
        raw_logits_chunk = flat_logits[start:end]
        logits_chunk = raw_logits_chunk.float() if needs_vocab_normalizer else raw_logits_chunk
        log_norm = torch.logsumexp(logits_chunk, dim=-1, keepdim=True) if needs_vocab_normalizer else None
        if topk_count is not None:
            top_logits, top_ids = torch.topk(logits_chunk, topk_count, dim=-1)
            topk_id_chunks.append(top_ids)
            if log_norm is None:
                raise RuntimeError("top-k log-prob computation requires a vocabulary normalizer.")
            topk_log_prob_chunks.append(top_logits - log_norm)
        if flat_gather_ids is not None:
            ids_chunk = flat_gather_ids[start:end]
            gathered_logits = logits_chunk.gather(dim=-1, index=ids_chunk)
            if normalize_gathered or logprob_mode == TOPK_LOGPROB_MODE_FULL_VOCAB:
                if log_norm is None:
                    raise RuntimeError("gathered log-prob computation requires a vocabulary normalizer.")
                gathered_logits = gathered_logits - log_norm
            gathered_log_prob_chunks.append(gathered_logits)

    topk_ids = None
    topk_log_probs = None
    gathered_log_probs = None
    if topk_count is not None:
        topk_ids = torch.cat(topk_id_chunks, dim=0).reshape(*prefix_shape, topk_count)
        topk_log_probs = torch.cat(topk_log_prob_chunks, dim=0).reshape(*prefix_shape, topk_count)
    if gather_shape is not None:
        gathered_log_probs = torch.cat(gathered_log_prob_chunks, dim=0).reshape(*prefix_shape, *gather_shape)
    return topk_ids, topk_log_probs, gathered_log_probs


def _log_tail_prob(top_log_probs: torch.Tensor) -> torch.Tensor:
    top_mass = torch.exp(top_log_probs).sum(dim=-1).clamp(min=0.0, max=1.0)
    tail_mass = (1.0 - top_mass).clamp(min=1e-12)
    return torch.log(tail_mass)


def _bucket_log_probs(top_log_probs: torch.Tensor, include_tail: bool) -> torch.Tensor:
    if not include_tail:
        return torch.log_softmax(top_log_probs, dim=-1)
    tail = _log_tail_prob(top_log_probs).unsqueeze(-1)
    return torch.cat([top_log_probs, tail], dim=-1)


def _temperature_bucket_log_probs(bucket_log_probs: torch.Tensor, temperature: float) -> torch.Tensor:
    if abs(temperature - 1.0) <= 1e-6:
        return bucket_log_probs
    return torch.log_softmax(bucket_log_probs / temperature, dim=-1)


def topk_distill_loss_matrix(
    *,
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    mode: str,
    include_tail: bool,
    temperature: float,
) -> torch.Tensor:
    """Return per-token KL loss on a shared top-k support.

    ``student_topk_log_probs`` and ``teacher_topk_log_probs`` must refer to
    the same selected token ids and have shape ``[batch, response, k]``.
    For renormalized support modes, these tensors only need to be log-scores
    on the selected support because ``log_softmax`` removes any global
    normalization constant.
    """

    if student_topk_log_probs.shape != teacher_topk_log_probs.shape:
        raise ValueError(
            "student_topk_log_probs and teacher_topk_log_probs must have identical shapes, "
            f"got {tuple(student_topk_log_probs.shape)} and {tuple(teacher_topk_log_probs.shape)}."
        )
    normalized_mode = str(mode)
    if normalized_mode not in TOPK_DISTILL_MODES:
        raise ValueError(f"Unsupported top-k distillation mode: {normalized_mode}")

    use_tail = include_tail and normalized_mode not in TOPK_RENORMALIZED_MODES
    teacher_log_q = _bucket_log_probs(teacher_topk_log_probs.float(), use_tail)
    student_log_q = _bucket_log_probs(student_topk_log_probs.float(), use_tail)
    teacher_log_q = _temperature_bucket_log_probs(teacher_log_q, temperature)
    student_log_q = _temperature_bucket_log_probs(student_log_q, temperature)

    teacher_q = torch.exp(teacher_log_q)
    student_q = torch.exp(student_log_q)
    if normalized_mode in TOPK_REVERSE_KL_MODES:
        return (student_q * (student_log_q - teacher_log_q)).sum(dim=-1)
    return (teacher_q * (teacher_log_q - student_log_q)).sum(dim=-1)


def topk_teacher_student_cross_entropy_matrix(
    *,
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    include_tail: bool,
    temperature: float,
) -> torch.Tensor:
    """Return per-token cross entropy H(p_teacher, p_student) on top-k buckets."""

    if student_topk_log_probs.shape != teacher_topk_log_probs.shape:
        raise ValueError(
            "student_topk_log_probs and teacher_topk_log_probs must have identical shapes, "
            f"got {tuple(student_topk_log_probs.shape)} and {tuple(teacher_topk_log_probs.shape)}."
        )
    teacher_log_q = _bucket_log_probs(teacher_topk_log_probs.float(), include_tail)
    student_log_q = _bucket_log_probs(student_topk_log_probs.float(), include_tail)
    teacher_log_q = _temperature_bucket_log_probs(teacher_log_q, temperature)
    student_log_q = _temperature_bucket_log_probs(student_log_q, temperature)
    teacher_q = torch.exp(teacher_log_q)
    return -(teacher_q * student_log_q).sum(dim=-1)


def topk_distill_bucket_metrics(
    *,
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    student_values_are_log_probs: bool = True,
    support_source: str = TOPK_SUPPORT_SOURCE_TEACHER,
) -> dict[str, float]:
    mask = response_mask.detach().float()
    denom = float(mask.sum().detach().cpu().item())
    if denom <= 0.0:
        return {}

    teacher_mass = torch.exp(teacher_topk_log_probs.detach().float()).sum(dim=-1).clamp(min=0.0, max=1.0)

    def masked_mean(value: torch.Tensor) -> float:
        return float(((value * mask).sum() / mask.sum().clamp(min=1.0)).detach().cpu().item())

    metrics = {
        "support_teacher_mass": masked_mean(teacher_mass),
        "tail_teacher_mass_off_support": masked_mean(1.0 - teacher_mass),
        "topk_teacher_mass": masked_mean(teacher_mass),
        "tail_teacher_mass": masked_mean(1.0 - teacher_mass),
    }
    if student_values_are_log_probs:
        student_mass = torch.exp(student_topk_log_probs.detach().float()).sum(dim=-1).clamp(min=0.0, max=1.0)
        metrics["support_student_mass"] = masked_mean(student_mass)
        metrics["tail_student_mass_off_support"] = masked_mean(1.0 - student_mass)
        if support_source == TOPK_SUPPORT_SOURCE_TEACHER:
            metrics["topk_student_mass_on_teacher_ids"] = metrics["support_student_mass"]
            metrics["tail_student_mass_on_teacher_ids"] = metrics["tail_student_mass_off_support"]
    return metrics
