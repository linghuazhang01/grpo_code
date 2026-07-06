# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig
from mopd_verl.full_gradient.actor_loss import build_actor_micro_batch_loss
from mopd_verl.topk_distill import (
    TOPK_LOGPROB_MODE_SPARSE,
    topk_distill_logprob_chunk_size,
    topk_distill_logprob_mode,
    topk_distill_uses_renormalized_support,
    topk_log_probs_from_logits,
    topk_teacher_student_cross_entropy_matrix,
)

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        calculate_entropy=False,
        inplace_backward: bool | None = None,
        topk: int | None = None,
        gather_topk_ids: torch.Tensor | None = None,
        calculate_log_probs: bool = True,
        normalize_gathered_topk: bool = True,
        topk_logprob_chunk_size: int | None = None,
        topk_logprob_mode: str = "sparse",
        return_extra: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        needs_topk_extra = return_extra or topk is not None or gather_topk_ids is not None
        if needs_topk_extra and self.use_fused_kernels:
            raise ValueError("Top-k distillation requires non-fused logits; set actor/ref use_fused_kernels=False.")
        if needs_topk_extra and self.use_ulysses_sp:
            raise ValueError("Top-k distillation is not supported with Ulysses sequence parallelism yet.")
        topk_ids = None
        topk_log_probs = None
        gathered_topk_log_probs = None
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            # reset input_ids, attention_mask, position_ids to ref model inputs if ref model input_ids is different from actor input_ids
            if "ref_input_ids" in micro_batch.keys():
                input_ids = micro_batch["ref_input_ids"]
                attention_mask = micro_batch["ref_attention_mask"]
                position_ids = micro_batch["ref_position_ids"]
                batch_size, seqlen = input_ids.shape

            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    if calculate_log_probs:
                        logprob_inplace_backward = True if inplace_backward is None else bool(inplace_backward)
                        if calculate_entropy or needs_topk_extra:
                            logprob_inplace_backward = False
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=logprob_inplace_backward,
                        )
                    else:
                        log_probs = logits_rmpad.new_zeros(input_ids_rmpad_rolled.shape)

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                    if needs_topk_extra:
                        gather_ids_rmpad = None
                        if gather_topk_ids is not None:
                            gather_ids = gather_topk_ids.to(device=input_ids.device, dtype=torch.long)
                            full_gather_ids = torch.zeros(
                                (batch_size, seqlen, int(gather_ids.shape[-1])),
                                device=input_ids.device,
                                dtype=torch.long,
                            )
                            full_gather_ids[:, -response_length - 1 : -1, :] = gather_ids
                            gather_ids_rmpad = index_first_axis(
                                rearrange(full_gather_ids, "b s k -> (b s) k"),
                                indices,
                            )
                        topk_ids_rmpad, topk_log_probs_rmpad, gathered_log_probs_rmpad = (
                            topk_log_probs_from_logits(
                                logits_rmpad,
                                topk=topk,
                                gather_topk_ids=gather_ids_rmpad,
                                normalize_gathered=normalize_gathered_topk,
                                chunk_size=topk_logprob_chunk_size or 16,
                                logprob_mode=topk_logprob_mode,
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )
                if topk is not None:
                    full_topk_log_probs = pad_input(
                        hidden_states=topk_log_probs_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    full_topk_ids = pad_input(
                        hidden_states=topk_ids_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    topk_log_probs = full_topk_log_probs[:, -response_length - 1 : -1, :]
                    topk_ids = full_topk_ids[:, -response_length - 1 : -1, :].long()
                if gather_topk_ids is not None:
                    full_gathered_topk_log_probs = pad_input(
                        hidden_states=gathered_log_probs_rmpad,
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    gathered_topk_log_probs = full_gathered_topk_log_probs[:, -response_length - 1 : -1, :]

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    if calculate_log_probs:
                        logprob_inplace_backward = True if inplace_backward is None else bool(inplace_backward)
                        if calculate_entropy or needs_topk_extra:
                            logprob_inplace_backward = False
                        log_probs = logprobs_from_logits(
                            logits,
                            micro_batch["responses"],
                            inplace_backward=logprob_inplace_backward,
                        )
                    else:
                        log_probs = logits.new_zeros(logits.shape[:-1])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    if needs_topk_extra:
                        topk_ids, topk_log_probs, gathered_topk_log_probs = topk_log_probs_from_logits(
                            logits,
                            topk=topk,
                            gather_topk_ids=gather_topk_ids,
                            normalize_gathered=normalize_gathered_topk,
                            chunk_size=topk_logprob_chunk_size or 16,
                            logprob_mode=topk_logprob_mode,
                        )

            if needs_topk_extra:
                return entropy, log_probs, topk_ids, topk_log_probs, gathered_topk_log_probs
            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(
        self,
        data: DataProto,
        calculate_entropy=False,
        topk: int | None = None,
        gather_topk_ids_key: str | None = None,
    ) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        has_ref_input_ids = "ref_input_ids" in data.batch.keys() # handle when ref input_ids is different from actor input_ids
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if gather_topk_ids_key is not None:
            select_keys.append(gather_topk_ids_key)
        if has_ref_input_ids:
            select_keys.extend(["ref_input_ids", "ref_attention_mask", "ref_position_ids"])
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        topk_ids_lst = []
        topk_log_probs_lst = []
        gathered_topk_log_probs_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            gather_topk_ids = (
                model_inputs[gather_topk_ids_key]
                if gather_topk_ids_key is not None
                else None
            )
            with torch.no_grad():
                forward_output = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    topk=topk,
                    gather_topk_ids=gather_topk_ids,
                    return_extra=topk is not None or gather_topk_ids is not None,
                )
                if topk is None and gather_topk_ids is None:
                    entropy, log_probs = forward_output
                else:
                    entropy, log_probs, topk_ids, topk_log_probs, gathered_topk_log_probs = forward_output
                    if topk_ids is not None:
                        topk_ids_lst.append(topk_ids)
                    if topk_log_probs is not None:
                        topk_log_probs_lst.append(topk_log_probs)
                    if gathered_topk_log_probs is not None:
                        gathered_topk_log_probs_lst.append(gathered_topk_log_probs)
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        topk_ids = torch.concat(topk_ids_lst, dim=0) if topk is not None else None
        topk_log_probs = torch.concat(topk_log_probs_lst, dim=0) if topk is not None else None
        gathered_topk_log_probs = (
            torch.concat(gathered_topk_log_probs_lst, dim=0)
            if gather_topk_ids_key is not None
            else None
        )

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if topk is not None:
                topk_ids = restore_dynamic_batch(topk_ids, batch_idx_list)
                topk_log_probs = restore_dynamic_batch(topk_log_probs, batch_idx_list)
            if gather_topk_ids_key is not None:
                gathered_topk_log_probs = restore_dynamic_batch(gathered_topk_log_probs, batch_idx_list)

        if topk is not None and gather_topk_ids_key is not None:
            return log_probs, entropys, topk_ids, topk_log_probs, gathered_topk_log_probs
        if topk is not None:
            return log_probs, entropys, topk_ids, topk_log_probs
        if gather_topk_ids_key is not None:
            return log_probs, entropys, gathered_topk_log_probs
        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor teacher-student cross entropy", logger=logger)
    def compute_teacher_student_cross_entropy(
        self,
        data: DataProto,
        *,
        teacher_topk_ids_key: str,
        teacher_topk_logprobs_key: str,
        include_tail: bool,
        distill_temperature: float,
    ) -> torch.Tensor:
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            teacher_topk_ids_key,
            teacher_topk_logprobs_key,
        ]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        cross_entropy_lst = []
        policy_loss_config = self.config.policy_loss
        use_renormalized_support = topk_distill_uses_renormalized_support(policy_loss_config)
        effective_topk_logprob_mode = topk_distill_logprob_mode(policy_loss_config)
        if use_renormalized_support:
            effective_topk_logprob_mode = TOPK_LOGPROB_MODE_SPARSE
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                _, _, _, _, student_topk_log_probs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    gather_topk_ids=model_inputs[teacher_topk_ids_key],
                    calculate_log_probs=False,
                    normalize_gathered_topk=not use_renormalized_support,
                    topk_logprob_chunk_size=topk_distill_logprob_chunk_size(policy_loss_config),
                    topk_logprob_mode=effective_topk_logprob_mode,
                    return_extra=True,
                )
                cross_entropy = topk_teacher_student_cross_entropy_matrix(
                    student_topk_log_probs=student_topk_log_probs,
                    teacher_topk_log_probs=model_inputs[teacher_topk_logprobs_key],
                    include_tail=include_tail,
                    temperature=distill_temperature,
                )
            cross_entropy_lst.append(cross_entropy)

        cross_entropy = torch.concat(cross_entropy_lst, dim=0)
        if use_dynamic_bsz:
            cross_entropy = restore_dynamic_batch(cross_entropy, batch_idx_list)
        return cross_entropy

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        batch_keys = set(data.batch.keys())

        def append_existing_batch_key(key: str) -> None:
            if key in batch_keys and key not in select_keys:
                select_keys.append(key)

        if self.config.use_kl_loss:
            select_keys.append("math_teacher_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        append_existing_batch_key("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        append_existing_batch_key("rollout_log_probs")
        # Include base model log probs for corrected reward computation
        # These are computed when actor_rollout_ref.model.base_model_path and
        # actor_rollout_ref.ref.model.base_model_path are both specified
        append_existing_batch_key("base_log_prob")
        append_existing_batch_key("code_teacher_log_prob")
        for key in (
            "math_teacher_topk_ids",
            "math_teacher_topk_logprobs",
            "code_teacher_topk_ids",
            "code_teacher_topk_logprobs",
            "student_topk_ids",
            "math_teacher_student_topk_logprobs",
            "code_teacher_student_topk_logprobs",
            "teacher_prefix_mask",
            "student_suffix_mask",
        ):
            append_existing_batch_key(key)
        # Include math_teacher_log_prob for only_reverse_kl_advantages mode
        teacher_prefix_config_active = bool(self.config.policy_loss.get("teacher_prefix_enabled", False))
        if (
            (self.config.policy_loss.only_reverse_kl_advantages or teacher_prefix_config_active)
            and "math_teacher_log_prob" in batch_keys
        ):
            append_existing_batch_key("math_teacher_log_prob")
        if teacher_prefix_config_active:
            append_existing_batch_key("code_teacher_log_prob")

        non_tensor_keys = set(data.non_tensor_batch.keys())
        has_multi_modal_inputs = "multi_modal_inputs" in non_tensor_keys
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        # Include audit/domain metadata. Training consumes opd_teacher; the
        # remaining keys are used by MOPD sample-level gradient logging.
        for key in ("opd_teacher", "sample_id", "id", "domain", "source_domain", "ability", "data_source", "extra_info"):
            if key in non_tensor_keys and key not in non_tensor_select_keys:
                non_tensor_select_keys.append(key)

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        # MOPD audit: domain-gradient tracker begin
        mopd_gradient_tracker = None
        mopd_full_gradient_cfg = data.meta_info.get("mopd_full_gradient", {})
        if not isinstance(mopd_full_gradient_cfg, dict):
            mopd_full_gradient_cfg = {}
        if isinstance(mopd_full_gradient_cfg, dict) and mopd_full_gradient_cfg.get("enabled", False):
            from mopd_verl.full_gradient_worker import SequentialBackwardDomainGradientTracker

            mopd_gradient_tracker = SequentialBackwardDomainGradientTracker(self, mopd_full_gradient_cfg)
        # MOPD audit: domain-gradient tracker end
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                mopd_static_micro_batch_size = None
                if (
                    mopd_gradient_tracker is not None
                    and bool(mopd_full_gradient_cfg.get("domain_gradient_enabled", False))
                ):
                    mopd_static_micro_batch_size = max(
                        1,
                        int(
                            mopd_full_gradient_cfg.get("micro_batch_size_per_gpu")
                            or self.config.ppo_micro_batch_size_per_gpu
                        ),
                    )
                if self.config.use_dynamic_bsz and mopd_static_micro_batch_size is not None:
                    metrics["global/audit/full_gradient_forced_static_micro_batch"] = 1.0
                    metrics["global/audit/full_gradient_static_micro_batch_size"] = float(
                        mopd_static_micro_batch_size
                    )
                use_dynamic_micro_batch = self.config.use_dynamic_bsz and mopd_static_micro_batch_size is None
                if use_dynamic_micro_batch:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, batch_idx_list = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batch_size = mopd_static_micro_batch_size or self.config.ppo_micro_batch_size_per_gpu
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // micro_batch_size
                    )
                    micro_batches = mini_batch.split(micro_batch_size)
                    batch_idx_list = []
                    start_idx = 0
                    for micro_batch in micro_batches:
                        end_idx = start_idx + len(micro_batch)
                        batch_idx_list.append(list(range(start_idx, end_idx)))
                        start_idx = end_idx
                micro_batches = list(micro_batches)
                if mopd_gradient_tracker is not None:
                    tracked_micro_batches = mopd_gradient_tracker.prepare_micro_batches(
                        micro_batches,
                        batch_idx_list=batch_idx_list,
                    )
                else:
                    tracked_micro_batches = [(None, micro_batch) for micro_batch in micro_batches]

                self.actor_optimizer.zero_grad()
                # MOPD audit: domain-gradient tracker begin
                if mopd_gradient_tracker is not None:
                    append_to_dict(
                        metrics,
                        mopd_gradient_tracker.run_pre_update_audit(
                            tracked_micro_batches,
                            on_policy=on_policy,
                            use_dynamic_micro_batch=use_dynamic_micro_batch,
                            ppo_mini_batch_size=self.config.ppo_mini_batch_size,
                            gradient_accumulation=self.gradient_accumulation,
                        ),
                    )
                    self.actor_optimizer.zero_grad()
                # MOPD audit: domain-gradient tracker end

                for _mopd_domain, micro_batch in tracked_micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]

                    if use_dynamic_micro_batch:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    loss_result = build_actor_micro_batch_loss(
                        self,
                        micro_batch,
                        loss_scale_factor=float(loss_scale_factor),
                        on_policy=on_policy,
                        include_metrics=True,
                        temperature=float(temperature),
                    )
                    loss = loss_result.loss
                    micro_batch_metrics.update(loss_result.metrics)
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    append_to_dict(metrics, micro_batch_metrics)

                if mopd_gradient_tracker is not None:
                    append_to_dict(
                        metrics,
                        mopd_gradient_tracker.full_grad_training_parity_metrics(),
                    )

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
