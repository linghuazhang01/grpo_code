"""Actor loss mirror used by full-gradient audit recompute paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mopd_verl.full_gradient.config import _cfg_get
from mopd_verl.full_gradient.labels import (
    _TEACHER_LABEL_KEY,
    _labels_from_mapping,
    _non_tensor_list,
)
from mopd_verl.topk_distill import (
    DISTILL_LOSS_BUILDER_POLICY_GRADIENT,
    TOPK_LOGPROB_MODE_SPARSE,
    TOPK_RENORMALIZED_FORWARD_KL,
    TOPK_SUPPORT_SOURCE_STUDENT,
    TOPK_SUPPORT_SOURCE_TEACHER,
    chosen_token_forward_kl_matrix,
    chosen_token_policy_gradient_reward_matrix,
    distill_loss_builder,
    resolved_topk_distill_mode,
    select_teacher_log_prob_tensor,
    teacher_prefix_forward_weight,
    teacher_prefix_masks,
    topk_distill_bucket_metrics,
    topk_distill_include_tail,
    topk_distill_logprob_chunk_size,
    topk_distill_logprob_mode,
    topk_distill_loss_matrix,
    topk_distill_support_source,
    topk_distill_temperature,
    topk_distill_uses_renormalized_support,
    topk_distill_weight,
    uses_topk_distill_loss,
)
from verl import DataProto
from verl.utils.device import get_device_id


@dataclass(frozen=True)
class ActorMicroBatchLossResult:
    loss: torch.Tensor
    metrics: dict[str, Any]


def _labels_from_inputs(model_inputs: dict[str, Any], batch_size: int) -> list[str]:
    return _labels_from_mapping(model_inputs, batch_size)


def _is_multi_teacher_distill_cfg(policy_loss_cfg: Any) -> bool:
    return bool(_cfg_get(policy_loss_cfg, "multi_teacher_distill", False))


def _topk_runtime_config(policy_loss_cfg: Any) -> tuple[bool, str]:
    use_renormalized_support = topk_distill_uses_renormalized_support(policy_loss_cfg)
    effective_topk_logprob_mode = (
        TOPK_LOGPROB_MODE_SPARSE
        if use_renormalized_support
        else topk_distill_logprob_mode(policy_loss_cfg)
    )
    return use_renormalized_support, effective_topk_logprob_mode


def _opd_teacher_labels_from_inputs(model_inputs: dict[str, Any], batch_size: int) -> list[Any]:
    """Return raw opd_teacher labels used by dp_actor.py for teacher selection.

    Domain-like labels such as ``domain``/``source_domain``/``ability`` are audit
    metadata. They must not influence teacher selection, otherwise recomputed
    gradients can diverge from the real actor loss.
    """
    return _non_tensor_list(model_inputs.get(_TEACHER_LABEL_KEY), batch_size)


def _code_teacher_mask_from_opd_teacher(
    model_inputs: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if _TEACHER_LABEL_KEY not in model_inputs:
        return torch.zeros(batch_size, dtype=torch.bool, device=device)

    labels = _opd_teacher_labels_from_inputs(model_inputs, batch_size)
    return torch.as_tensor(
        [label == "code" for label in labels],
        dtype=torch.bool,
        device=device,
    )


def _select_by_code_teacher(
    math_tensor: torch.Tensor,
    code_tensor: torch.Tensor | None,
    model_inputs: dict[str, Any],
    policy_loss_cfg: Any,
) -> torch.Tensor:
    if (
        code_tensor is None
        or not _is_multi_teacher_distill_cfg(policy_loss_cfg)
        or _TEACHER_LABEL_KEY not in model_inputs
    ):
        return math_tensor

    code_mask = _code_teacher_mask_from_opd_teacher(
        model_inputs,
        batch_size=int(math_tensor.shape[0]),
        device=math_tensor.device,
    )
    if not bool(code_mask.any()):
        return math_tensor
    if bool(code_mask.all()):
        return code_tensor

    view_shape = (int(code_mask.shape[0]),) + (1,) * (math_tensor.dim() - 1)
    return torch.where(code_mask.view(view_shape), code_tensor, math_tensor)


def _selected_teacher_log_prob_from_inputs(
    model_inputs: dict[str, Any],
    policy_loss_cfg: Any,
) -> torch.Tensor:
    if "math_teacher_log_prob" not in model_inputs:
        raise ValueError("Reverse-KL advantages require math_teacher_log_prob in the batch.")
    return _select_by_code_teacher(
        math_tensor=model_inputs["math_teacher_log_prob"],
        code_tensor=model_inputs.get("code_teacher_log_prob"),
        model_inputs=model_inputs,
        policy_loss_cfg=policy_loss_cfg,
    )


def _selected_teacher_topk_from_inputs(
    model_inputs: dict[str, Any],
    policy_loss_cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "math_teacher_topk_ids" not in model_inputs or "math_teacher_topk_logprobs" not in model_inputs:
        raise ValueError(
            "Top-k distillation requires math_teacher_topk_ids and math_teacher_topk_logprobs in the batch."
        )
    math_ids = model_inputs["math_teacher_topk_ids"]
    math_log_probs = model_inputs["math_teacher_topk_logprobs"]
    if (
        not _is_multi_teacher_distill_cfg(policy_loss_cfg)
        or "code_teacher_topk_ids" not in model_inputs
    ):
        return math_ids, math_log_probs
    return (
        _select_by_code_teacher(
            math_tensor=math_ids,
            code_tensor=model_inputs["code_teacher_topk_ids"],
            model_inputs=model_inputs,
            policy_loss_cfg=policy_loss_cfg,
        ),
        _select_by_code_teacher(
            math_tensor=math_log_probs,
            code_tensor=model_inputs.get("code_teacher_topk_logprobs", math_log_probs),
            model_inputs=model_inputs,
            policy_loss_cfg=policy_loss_cfg,
        ),
    )


def _selected_student_topk_teacher_log_probs(
    model_inputs: dict[str, Any],
    policy_loss_cfg: Any,
) -> torch.Tensor:
    if "math_teacher_student_topk_logprobs" not in model_inputs:
        raise ValueError(
            "Student top-k distillation requires math_teacher_student_topk_logprobs in the batch."
        )
    math_log_probs = model_inputs["math_teacher_student_topk_logprobs"]
    if (
        not _is_multi_teacher_distill_cfg(policy_loss_cfg)
        or "code_teacher_student_topk_logprobs" not in model_inputs
    ):
        return math_log_probs
    return _select_by_code_teacher(
        math_tensor=math_log_probs,
        code_tensor=model_inputs["code_teacher_student_topk_logprobs"],
        model_inputs=model_inputs,
        policy_loss_cfg=policy_loss_cfg,
    )


def _selected_topk_support_from_inputs(
    model_inputs: dict[str, Any],
    policy_loss_cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    support_source = topk_distill_support_source(policy_loss_cfg)
    if support_source == TOPK_SUPPORT_SOURCE_STUDENT:
        if "student_topk_ids" not in model_inputs:
            raise ValueError("Student top-k distillation requires student_topk_ids in the batch.")
        return model_inputs["student_topk_ids"], _selected_student_topk_teacher_log_probs(
            model_inputs,
            policy_loss_cfg,
        )
    if support_source != TOPK_SUPPORT_SOURCE_TEACHER:
        raise ValueError(f"Unsupported top-k support source: {support_source!r}.")
    return _selected_teacher_topk_from_inputs(model_inputs, policy_loss_cfg)


def _actor_reverse_kl_advantages(
    actor: Any,
    model_inputs: dict[str, Any],
    old_log_prob: torch.Tensor,
) -> torch.Tensor:
    """Mirror dp_actor.py's only_reverse_kl_advantages branch exactly."""
    policy_loss_cfg = _cfg_get(actor.config, "policy_loss", {})
    if not bool(_cfg_get(policy_loss_cfg, "only_reverse_kl_advantages", False)):
        return model_inputs["advantages"]

    math_teacher_log_prob = model_inputs["math_teacher_log_prob"]
    base_log_prob = model_inputs.get("base_log_prob")
    lambda_vals = float(_cfg_get(policy_loss_cfg, "lambda_vals", 1.0))
    multi_teacher = _is_multi_teacher_distill_cfg(policy_loss_cfg)

    if base_log_prob is not None:
        if multi_teacher:
            if _TEACHER_LABEL_KEY in model_inputs:
                teacher_log_prob = _selected_teacher_log_prob_from_inputs(
                    model_inputs,
                    policy_loss_cfg,
                )
                if lambda_vals == 1.0:
                    reverse_kl = old_log_prob - teacher_log_prob
                else:
                    reverse_kl = (old_log_prob - base_log_prob) - (
                        teacher_log_prob - base_log_prob
                    ) * lambda_vals
            else:
                # Keep dp_actor.py semantics: multi-teacher without opd_teacher
                # falls back to math teacher and skips base correction.
                reverse_kl = old_log_prob - math_teacher_log_prob
        else:
            if lambda_vals == 1.0:
                reverse_kl = old_log_prob - math_teacher_log_prob
            else:
                reverse_kl = (old_log_prob - base_log_prob) - (
                    math_teacher_log_prob - base_log_prob
                ) * lambda_vals

    elif multi_teacher and _TEACHER_LABEL_KEY in model_inputs and "code_teacher_log_prob" in model_inputs:
        teacher_log_prob = _selected_teacher_log_prob_from_inputs(
            model_inputs,
            policy_loss_cfg,
        )
        reverse_kl = old_log_prob - teacher_log_prob

    else:
        reverse_kl = old_log_prob - math_teacher_log_prob

    return -reverse_kl


def _actor_policy_gradient_rewards(
    model_inputs: dict[str, Any],
    policy_loss_cfg: Any,
    old_log_prob: torch.Tensor,
) -> torch.Tensor:
    teacher_log_prob = _selected_teacher_log_prob_from_inputs(model_inputs, policy_loss_cfg)
    return chosen_token_policy_gradient_reward_matrix(
        student_log_probs=old_log_prob,
        teacher_log_probs=teacher_log_prob,
    )


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float:
    denom = mask.detach().float().sum().clamp(min=1.0)
    return float(((value.detach().float() * mask.detach().float()).sum() / denom).cpu().item())


def _response_token_id_matrix_from_inputs(model_inputs: dict[str, Any], response_mask: torch.Tensor) -> torch.Tensor | None:
    token_ids = None
    for key in ("responses", "response_ids", "input_ids"):
        if key in model_inputs:
            token_ids = model_inputs[key]
            break
    if token_ids is None or not hasattr(token_ids, "detach") or len(token_ids.shape) != 2:
        return None
    response_len = int(response_mask.shape[-1])
    if tuple(token_ids.shape) == tuple(response_mask.shape):
        return token_ids.detach().long()
    if int(token_ids.shape[0]) == int(response_mask.shape[0]) and int(token_ids.shape[-1]) >= response_len:
        return token_ids[:, -response_len:].detach().long()
    return None


def _data_proto_tensor_device(data: DataProto) -> torch.device | None:
    if data.batch is None:
        return None
    for tensor in data.batch.values():
        if hasattr(tensor, "device"):
            return tensor.device
    return None


def _copy_data_proto_rows_to_cpu(data: DataProto, indices: list[int]) -> DataProto | None:
    if not indices:
        return None
    try:
        device = _data_proto_tensor_device(data)
        if device is None:
            idxs = torch.tensor(indices, dtype=torch.long)
        else:
            idxs = torch.tensor(indices, dtype=torch.long, device=device)
        return data.select_idxs(idxs).to("cpu")
    except Exception:
        return None


def _token_contribution_scale(response_mask: torch.Tensor, sample_idx: int, loss_agg_mode: str) -> float:
    active_tokens_total = float(response_mask.detach().sum().item())
    active_tokens_sample = float(response_mask.detach()[sample_idx].sum().item())
    active_sequences = float((response_mask.detach().sum(dim=-1) > 0).float().sum().item())
    if active_tokens_sample <= 0.0:
        return 0.0
    if loss_agg_mode == "token-mean":
        return 0.0 if active_tokens_total <= 0.0 else 1.0 / active_tokens_total
    if loss_agg_mode == "seq-mean-token-sum":
        return 0.0 if active_sequences <= 0.0 else 1.0 / active_sequences
    if loss_agg_mode == "seq-mean-token-mean":
        return 0.0 if active_sequences <= 0.0 else 1.0 / (active_sequences * active_tokens_sample)
    if loss_agg_mode == "seq-mean-token-sum-norm":
        return 1.0 / max(int(response_mask.shape[-1]), 1)
    return 0.0 if active_tokens_total <= 0.0 else 1.0 / active_tokens_total


def _token_mask_contribution_scale(
    response_mask: torch.Tensor,
    token_mask: torch.Tensor,
    loss_agg_mode: str,
) -> float:
    response_mask = response_mask.detach().float()
    token_mask = token_mask.detach().float() * response_mask
    selected_tokens = float(token_mask.sum().item())
    if selected_tokens <= 0.0:
        return 0.0
    active_tokens_total = float(response_mask.sum().item())
    if loss_agg_mode == "token-mean":
        return 0.0 if active_tokens_total <= 0.0 else selected_tokens / active_tokens_total
    active_sequences = float((response_mask.sum(dim=-1) > 0).float().sum().item())
    selected_sequences = float((token_mask.sum(dim=-1) > 0).float().sum().item())
    if loss_agg_mode == "seq-mean-token-sum":
        return 0.0 if active_sequences <= 0.0 else selected_sequences / active_sequences
    if loss_agg_mode == "seq-mean-token-mean":
        total = 0.0
        for sample_idx in range(int(response_mask.shape[0])):
            sample_selected = float(token_mask[sample_idx].sum().item())
            sample_tokens = float(response_mask[sample_idx].sum().item())
            if sample_selected > 0.0 and sample_tokens > 0.0:
                total += sample_selected / sample_tokens
        return 0.0 if active_sequences <= 0.0 else total / active_sequences
    if loss_agg_mode == "seq-mean-token-sum-norm":
        return selected_tokens / max(int(response_mask.shape[-1]), 1)
    return 0.0 if active_tokens_total <= 0.0 else selected_tokens / active_tokens_total


def build_actor_micro_batch_loss(
    actor: Any,
    micro_batch: DataProto,
    *,
    loss_scale_factor: float,
    on_policy: bool,
    safe_logprob_backward: bool = False,
    response_mask_override: torch.Tensor | None = None,
    include_metrics: bool = False,
    temperature: float | None = None,
) -> ActorMicroBatchLossResult:
    from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty

    micro_batch = micro_batch.to(get_device_id())
    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
    metrics: dict[str, Any] = {}
    response_mask = model_inputs["response_mask"]
    entropy_coeff = float(_cfg_get(actor.config, "entropy_coeff", 0.0) or 0.0)
    forward_temperature = (
        float(temperature) if temperature is not None else float(micro_batch.meta_info.get("temperature", 1.0))
    )
    forward_kwargs = {
        "temperature": forward_temperature,
        "calculate_entropy": entropy_coeff != 0,
    }
    if safe_logprob_backward:
        forward_kwargs["inplace_backward"] = False
    policy_loss_cfg = _cfg_get(actor.config, "policy_loss", {})
    builder_name = distill_loss_builder(policy_loss_cfg)
    topk_distill_active = uses_topk_distill_loss(policy_loss_cfg)
    use_renormalized_support, effective_topk_logprob_mode = _topk_runtime_config(policy_loss_cfg)
    use_renormalized_support = topk_distill_active and use_renormalized_support
    kl_coef = float(_cfg_get(actor.config, "kl_loss_coef", 0.0) or 0.0)
    needs_log_probs = not topk_distill_active or (
        bool(_cfg_get(actor.config, "use_kl_loss", False)) and kl_coef != 0.0
    )
    forward_kwargs["calculate_log_probs"] = needs_log_probs
    topk_support_ids = None
    teacher_support_log_probs = None
    topk_support_source_value = topk_distill_support_source(policy_loss_cfg)
    if topk_distill_active:
        topk_support_ids, teacher_support_log_probs = _selected_topk_support_from_inputs(
            model_inputs,
            policy_loss_cfg,
        )
        forward_kwargs["gather_topk_ids"] = topk_support_ids
        forward_kwargs["normalize_gathered_topk"] = not use_renormalized_support
        forward_kwargs["topk_logprob_chunk_size"] = topk_distill_logprob_chunk_size(policy_loss_cfg)
        forward_kwargs["topk_logprob_mode"] = effective_topk_logprob_mode
        forward_kwargs["return_extra"] = True
    forward_output = actor._forward_micro_batch(model_inputs, **forward_kwargs)
    if topk_distill_active:
        entropy, log_prob, _topk_ids, _topk_log_probs, student_topk_log_probs = forward_output
    else:
        entropy, log_prob = forward_output
    if response_mask_override is not None:
        base_response_mask = response_mask.to(device=log_prob.device, dtype=response_mask.dtype)
        response_mask = response_mask_override.to(device=log_prob.device, dtype=response_mask.dtype)
        response_mask = response_mask * base_response_mask
        model_inputs = dict(model_inputs)
        model_inputs["response_mask"] = response_mask
        for mask_key in ("teacher_prefix_mask", "student_suffix_mask"):
            if mask_key in model_inputs:
                model_inputs[mask_key] = (
                    model_inputs[mask_key].to(device=log_prob.device, dtype=response_mask.dtype)
                    * response_mask
                )
    prefix_loss_mask, suffix_loss_mask, teacher_prefix_active = teacher_prefix_masks(
        model_inputs,
        response_mask,
        policy_loss_cfg,
    )
    if response_mask_override is not None:
        prefix_loss_mask = prefix_loss_mask * response_mask
        suffix_loss_mask = suffix_loss_mask * response_mask
    distill_response_mask = suffix_loss_mask if teacher_prefix_active else response_mask
    loss_token_mask = (
        (prefix_loss_mask + suffix_loss_mask).clamp(max=1.0)
        if teacher_prefix_active
        else response_mask
    )
    if response_mask_override is not None and float(response_mask.detach().sum().item()) <= 0.0:
        zero_source = student_topk_log_probs if topk_distill_active else log_prob
        zero_loss = zero_source.sum() * 0.0
        if include_metrics:
            metrics["actor/pg_loss"] = 0.0
        return ActorMicroBatchLossResult(loss=zero_loss, metrics=metrics)
    if bool(_cfg_get(actor.config, "use_rollout_log_probs", False)):
        old_log_prob = model_inputs["old_log_probs"]
    elif on_policy:
        old_log_prob = log_prob.detach()
    else:
        old_log_prob = model_inputs["old_log_probs"]

    if topk_distill_active:
        policy_loss = log_prob.new_zeros(())
        pg_loss = policy_loss
    else:
        if builder_name == DISTILL_LOSS_BUILDER_POLICY_GRADIENT:
            advantages = _actor_policy_gradient_rewards(model_inputs, policy_loss_cfg, old_log_prob)
        else:
            advantages = _actor_reverse_kl_advantages(actor, model_inputs, old_log_prob)
        loss_mode = str(_cfg_get(_cfg_get(actor.config, "policy_loss", {}), "loss_mode", "vanilla"))
        policy_loss_fn = get_policy_loss_fn(loss_mode)
        pg_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=distill_response_mask,
            loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
            config=actor.config,
            rollout_is_weights=model_inputs.get("rollout_is_weights", None),
        )
        policy_loss = pg_loss
        if include_metrics:
            metrics.update(pg_metrics)
            if builder_name == DISTILL_LOSS_BUILDER_POLICY_GRADIENT:
                metrics["actor/chosen_token_pg_reward_mean"] = _masked_mean(
                    advantages,
                    distill_response_mask,
                )
            rollout_log_prob = model_inputs.get("rollout_log_probs", None)
            if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                metrics.update(
                    compute_rollout_corr_metrics_from_logprobs(
                        log_prob=log_prob,
                        rollout_log_prob=rollout_log_prob,
                        response_mask=distill_response_mask,
                    )
                )
    if entropy_coeff != 0 and entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy,
            loss_mask=loss_token_mask,
            loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
        )
        policy_loss = policy_loss - entropy_loss * entropy_coeff
    if topk_distill_active:
        topk_loss_mat = topk_distill_loss_matrix(
            student_topk_log_probs=student_topk_log_probs,
            teacher_topk_log_probs=teacher_support_log_probs,
            mode=resolved_topk_distill_mode(policy_loss_cfg),
            include_tail=topk_distill_include_tail(policy_loss_cfg),
            temperature=topk_distill_temperature(policy_loss_cfg),
        )
        topk_loss = agg_loss(
            loss_mat=topk_loss_mat,
            loss_mask=distill_response_mask,
            loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
        )
        topk_weight = topk_distill_weight(policy_loss_cfg)
        policy_loss = policy_loss + topk_loss * topk_weight
        if include_metrics:
            metrics["actor/topk_distill_loss"] = topk_loss.detach().item() * float(loss_scale_factor)
            metrics["actor/topk_distill_weight"] = topk_weight
            metrics["actor/topk_distill_support_is_student"] = float(
                topk_support_source_value == TOPK_SUPPORT_SOURCE_STUDENT
            )
            for key, value in topk_distill_bucket_metrics(
                student_topk_log_probs=student_topk_log_probs,
                teacher_topk_log_probs=teacher_support_log_probs,
                response_mask=distill_response_mask,
                student_values_are_log_probs=not use_renormalized_support,
                support_source=topk_support_source_value,
            ).items():
                metrics[f"actor/{key}"] = value
    if teacher_prefix_active:
        prefix_weight = teacher_prefix_forward_weight(policy_loss_cfg)
        if topk_distill_active:
            prefix_loss_mat = topk_distill_loss_matrix(
                student_topk_log_probs=student_topk_log_probs,
                teacher_topk_log_probs=teacher_support_log_probs,
                mode=TOPK_RENORMALIZED_FORWARD_KL,
                include_tail=False,
                temperature=topk_distill_temperature(policy_loss_cfg),
            )
        else:
            teacher_log_prob = select_teacher_log_prob_tensor(model_inputs, policy_loss_cfg)
            prefix_loss_mat = chosen_token_forward_kl_matrix(
                student_log_probs=log_prob,
                teacher_log_probs=teacher_log_prob,
            )
        prefix_loss = agg_loss(
            loss_mat=prefix_loss_mat,
            loss_mask=prefix_loss_mask,
            loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
        )
        policy_loss = policy_loss + prefix_loss * prefix_weight
        if include_metrics:
            metrics["actor/teacher_prefix_forward_kl_loss"] = (
                prefix_loss.detach().item() * float(loss_scale_factor)
            )
            metrics["actor/teacher_prefix_forward_kl_weight"] = prefix_weight
            metrics["actor/teacher_prefix_token_count"] = (
                prefix_loss_mask.detach().sum().item() * float(loss_scale_factor)
            )
            metrics["actor/student_suffix_token_count"] = (
                suffix_loss_mask.detach().sum().item() * float(loss_scale_factor)
            )
    if bool(_cfg_get(actor.config, "use_kl_loss", False)) and "math_teacher_log_prob" in model_inputs:
        if kl_coef != 0:
            kld = kl_penalty(
                logprob=log_prob,
                ref_logprob=model_inputs["math_teacher_log_prob"],
                kl_penalty=str(_cfg_get(actor.config, "kl_loss_type", "kl")),
            )
            kl_loss = agg_loss(
                loss_mat=kld,
                loss_mask=distill_response_mask,
                loss_agg_mode=str(_cfg_get(actor.config, "loss_agg_mode", "token-mean")),
            )
            policy_loss = policy_loss + kl_loss * kl_coef
            if include_metrics:
                metrics["actor/kl_loss"] = kl_loss.detach().item() * float(loss_scale_factor)
                metrics["actor/kl_coef"] = kl_coef
    if include_metrics:
        metrics["actor/pg_loss"] = pg_loss.detach().item() * float(loss_scale_factor)
    return ActorMicroBatchLossResult(
        loss=policy_loss * float(loss_scale_factor),
        metrics=metrics,
    )


def _actor_micro_batch_loss(
    actor: Any,
    micro_batch: DataProto,
    *,
    loss_scale_factor: float,
    on_policy: bool,
    safe_logprob_backward: bool = False,
    response_mask_override: torch.Tensor | None = None,
) -> torch.Tensor:
    return build_actor_micro_batch_loss(
        actor,
        micro_batch,
        loss_scale_factor=loss_scale_factor,
        on_policy=on_policy,
        safe_logprob_backward=safe_logprob_backward,
        response_mask_override=response_mask_override,
        include_metrics=False,
    ).loss


def _actor_micro_batch_token_loss_scores(
    actor: Any,
    micro_batch: DataProto,
    *,
    on_policy: bool,
) -> tuple[torch.Tensor | None, str]:
    """Return a detached per-response-token loss score matrix for token selection."""

    from verl.trainer.ppo.core_algos import kl_penalty

    try:
        with torch.no_grad():
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            response_mask = model_inputs["response_mask"]
            policy_loss_cfg = _cfg_get(actor.config, "policy_loss", {})
            topk_distill_active = uses_topk_distill_loss(policy_loss_cfg)
            builder_name = distill_loss_builder(policy_loss_cfg)
            use_renormalized_support, effective_topk_logprob_mode = _topk_runtime_config(policy_loss_cfg)
            use_renormalized_support = topk_distill_active and use_renormalized_support
            kl_coef = float(_cfg_get(actor.config, "kl_loss_coef", 0.0) or 0.0)
            needs_log_probs = not topk_distill_active or (
                bool(_cfg_get(actor.config, "use_kl_loss", False)) and kl_coef != 0.0
            )
            forward_kwargs = {
                "temperature": float(micro_batch.meta_info.get("temperature", 1.0)),
                "calculate_entropy": False,
                "calculate_log_probs": needs_log_probs,
            }
            topk_support_ids = None
            teacher_support_log_probs = None
            if topk_distill_active:
                topk_support_ids, teacher_support_log_probs = _selected_topk_support_from_inputs(
                    model_inputs,
                    policy_loss_cfg,
                )
                forward_kwargs["gather_topk_ids"] = topk_support_ids
                forward_kwargs["normalize_gathered_topk"] = not use_renormalized_support
                forward_kwargs["topk_logprob_chunk_size"] = topk_distill_logprob_chunk_size(policy_loss_cfg)
                forward_kwargs["topk_logprob_mode"] = effective_topk_logprob_mode
                forward_kwargs["return_extra"] = True

            forward_output = actor._forward_micro_batch(model_inputs, **forward_kwargs)
            if topk_distill_active:
                _entropy, log_prob, _topk_ids, _topk_log_probs, student_topk_log_probs = forward_output
            else:
                _entropy, log_prob = forward_output

            prefix_loss_mask, suffix_loss_mask, teacher_prefix_active = teacher_prefix_masks(
                model_inputs,
                response_mask,
                policy_loss_cfg,
            )
            distill_response_mask = suffix_loss_mask if teacher_prefix_active else response_mask

            if topk_distill_active:
                loss_mat = topk_distill_loss_matrix(
                    student_topk_log_probs=student_topk_log_probs,
                    teacher_topk_log_probs=teacher_support_log_probs,
                    mode=resolved_topk_distill_mode(policy_loss_cfg),
                    include_tail=topk_distill_include_tail(policy_loss_cfg),
                    temperature=topk_distill_temperature(policy_loss_cfg),
                )
                loss_mat = loss_mat * topk_distill_weight(policy_loss_cfg)
                source = "topk_distill_loss"
            else:
                if bool(_cfg_get(actor.config, "use_rollout_log_probs", False)):
                    old_log_prob = model_inputs["old_log_probs"]
                elif on_policy:
                    old_log_prob = log_prob.detach()
                else:
                    old_log_prob = model_inputs["old_log_probs"]
                if builder_name == DISTILL_LOSS_BUILDER_POLICY_GRADIENT:
                    loss_mat = _actor_policy_gradient_rewards(
                        model_inputs,
                        policy_loss_cfg,
                        old_log_prob,
                    )
                    source = "chosen_token_policy_gradient_reward"
                else:
                    teacher_log_prob = _selected_teacher_log_prob_from_inputs(model_inputs, policy_loss_cfg)
                    loss_mat = old_log_prob.float() - teacher_log_prob.float()
                    source = "chosen_token_reverse_kl_proxy"

            score_mat = loss_mat.float() * distill_response_mask.detach().float()
            if teacher_prefix_active and prefix_loss_mask.detach().sum().item() > 0:
                if topk_distill_active:
                    prefix_loss_mat = topk_distill_loss_matrix(
                        student_topk_log_probs=student_topk_log_probs,
                        teacher_topk_log_probs=teacher_support_log_probs,
                        mode=TOPK_RENORMALIZED_FORWARD_KL,
                        include_tail=False,
                        temperature=topk_distill_temperature(policy_loss_cfg),
                    )
                else:
                    teacher_log_prob = _selected_teacher_log_prob_from_inputs(model_inputs, policy_loss_cfg)
                    prefix_loss_mat = chosen_token_forward_kl_matrix(
                        student_log_probs=log_prob,
                        teacher_log_probs=teacher_log_prob,
                    )
                score_mat = (
                    score_mat
                    + prefix_loss_mat.float()
                    * prefix_loss_mask.detach().float()
                    * teacher_prefix_forward_weight(policy_loss_cfg)
                )

            if bool(_cfg_get(actor.config, "use_kl_loss", False)) and kl_coef != 0.0 and "math_teacher_log_prob" in model_inputs:
                kld = kl_penalty(
                    logprob=log_prob,
                    ref_logprob=model_inputs["math_teacher_log_prob"],
                    kl_penalty=str(_cfg_get(actor.config, "kl_loss_type", "kl")),
                )
                score_mat = score_mat + kld.float() * distill_response_mask.detach().float() * kl_coef

            return score_mat.detach().float().cpu(), source
    except Exception as exc:
        return None, f"unavailable_{type(exc).__name__}"
