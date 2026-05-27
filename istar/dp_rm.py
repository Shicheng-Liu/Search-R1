# Copyright 2024 PRIME team and/or its affiliates
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
Implement a multiprocess PPOCritic
"""

import itertools
from typing import Dict, Tuple, Optional

import torch
import torch.distributed
from flash_attn.bert_padding import (index_first_axis, pad_input, rearrange,
                                     unpad_input)
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from istar.rm_utils import (compute_ce_dpo_loss_rm, compute_bt_loss_rm, compute_irl_loss_rm,
                          compute_detach_dpo_loss_rm, compute_eto_loss_rm, composite_rm_loss, compute_ce_hyper_loss_rm, compute_bt_hyper_loss_rm)
from verl import DataProto
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import (get_reverse_idx,
                                         rearrange_micro_batches)
from verl.utils.ulysses import (gather_outpus_and_unpad,
                                ulysses_pad_and_slice_inputs)

__all__ = ["DataParallelPRIMERewardModel"]


def tensor_grad_norm_from_loss(loss, tensor, retain_graph=True):
    grad = torch.autograd.grad(
        loss,
        tensor,
        retain_graph=retain_graph,
        allow_unused=True,
    )[0]

    if grad is None:
        return 0.0, 0.0, 0.0

    grad = grad.detach().float()
    return (
        grad.norm(2).item(),
        grad.abs().mean().item(),
        grad.abs().max().item(),
    )

def print_param_update(model, tag):
    with torch.no_grad():
        total_norm = 0.0
        total_grad = 0.0
        for name, p in model.named_parameters():
            if p.requires_grad:
                total_norm += p.data.float().norm().item()
                if p.grad is not None:
                    total_grad += p.grad.data.float().norm().item()
        print(f"[{tag}] param_norm={total_norm:.6e}, grad_norm={total_grad:.6e}")

class DataParallelISTARRewardModel:
    def __init__(self, config, reward_module: nn.Module, ref_module: nn.Module, reward_optimizer: optim.Optimizer):
        self.config = config
        self.reward_module = reward_module
        self.ref_module = ref_module
        self.reward_optimizer = reward_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Reward model use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.model.get("use_fused_kernels", False)
        print(f"Reward model use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)

    def _forward_micro_batch(self, micro_batch, prompt_length):
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]

        num_actions = micro_batch["input_ids"].shape[-1] - prompt_length
        max_positions = micro_batch["attention_mask"][:, prompt_length:].sum(-1)

        if self.use_remove_padding:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size)

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)
            output = self.reward_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                use_cache=False,
            )

            if self.use_fused_kernels:
                rm_log_labels = output.log_probs.squeeze(0)  # (total_nnz,)
                rm_log_labels = rm_log_labels.to(torch.float32)

            else:
                rm_output_logits = output.logits.squeeze(0)
                rm_log_labels = verl_F.logprobs_from_logits(
                    logits=rm_output_logits,
                    labels=input_ids_rmpad_rolled,
                )

            if self.ulysses_sequence_parallel_size > 1:
                rm_log_labels = gather_outpus_and_unpad(rm_log_labels, gather_dim=0, unpad_dim=0, padding_size=pad_size)
            rm_log_labels = pad_input(hidden_states=rm_log_labels.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)[:, -num_actions - 1 : -1]

        else:
            output = self.reward_module(
                input_ids=micro_batch["input_ids"],
                attention_mask=micro_batch["attention_mask"],
                position_ids=micro_batch["position_ids"],
                use_cache=False,
            )

            if self.use_fused_kernels:
                rm_log_labels = output.log_probs[:, :-1]  # (bsz, seq_length)
                rm_log_labels = rm_log_labels.to(torch.float32)

            else:
                rm_output_logits = output.logits
                rm_log_labels = verl_F.logprobs_from_logits(logits=rm_output_logits[:, :-1, :],labels=micro_batch["input_ids"][:, 1:])
                # rm_log_prob = torch.nn.functional.log_softmax(rm_output_logits[:, :-1, :], dim=-1)  # (batch_size, seq_length, vocab_size)
                # rm_log_labels = rm_log_prob.gather(dim=-1, index=micro_batch["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)  # (batch, seq_length)

        if self.ref_module is not None:
            # do not have to pad again
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if self.ulysses_sequence_parallel_size > 1 and self.use_remove_padding:
                    ref_output = self.ref_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        use_cache=False,
                    )

                    if self.use_fused_kernels:
                        ref_log_labels = ref_output.log_probs.squeeze(0)  # (total_nnz,)
                        ref_log_labels = ref_log_labels.to(torch.float32)

                    else:
                        ref_output_logits = ref_output.logits.squeeze(0)
                        ref_log_labels = verl_F.logprobs_from_logits(logits=ref_output_logits, labels=input_ids_rmpad_rolled)

                    ref_log_labels = gather_outpus_and_unpad(ref_log_labels, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    ref_log_labels = pad_input(hidden_states=ref_log_labels.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)[:, -num_actions - 1 : -1]
                else:
                    ref_output = self.ref_module(
                        input_ids=micro_batch["input_ids"],
                        attention_mask=micro_batch["attention_mask"],
                        position_ids=micro_batch["position_ids"],
                        use_cache=False,
                    )

                    if self.use_fused_kernels:
                        ref_log_labels = ref_output.log_probs[:, :-1]  # (batch_size, seq_length)
                        ref_log_labels = ref_log_labels.to(torch.float32)

                    else:
                        ref_output_logits = ref_output.logits
                        ref_log_labels = verl_F.logprobs_from_logits(logits=ref_output_logits[:, :-1, :], labels=micro_batch["input_ids"][:, 1:])
                        # ref_log_prob = torch.nn.functional.log_softmax(ref_output_logits[:, :-1, :], dim=-1)  # (batch_size, seq_length, vocab_size)
                        # ref_log_labels = ref_log_prob.gather(dim=-1, index=micro_batch["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)  # (batch, seq_length)

        else:
            ref_log_labels = micro_batch["old_log_probs"]

        ref_log_labels.to(rm_log_labels.dtype)
        q = rm_log_labels[:, -num_actions:] - ref_log_labels[:, -num_actions:]  # this is actually diff of q    

        # trim unnecessary logprobs here
        for i in range(micro_batch["input_ids"].shape[0]):
            q[i, max_positions[i] :] = 0

        # reward computation does not need gradient. only q needs
        with torch.no_grad():
            # generalized estimation of r should go before the reward filling. r means process reward for policy model, or the advantage of reward model.
            lam = self.config.get("lambda", 0.0)
            beta = self.config.model.get("beta_train", 0.05)
            if lam == 0.0:
                r = q * beta
            else:
                # reward coefficient takes no effect here
                acc = micro_batch["acc"]
                q_ = q * beta
                r = torch.zeros_like(q)
                lastgaelam = 0
                # change the last token and mask out all paddings to make this process easier if we rely on outcome reward to calculate V
                for i in range(q.shape[0]):
                    if self.config.prime_use_gt:
                        q_[i, max_positions[i] - 1] = acc[i] - q_[i, : max_positions[i] - 1].sum()
                    q_[i, max_positions[i] :] = 0

                for t in reversed(range(num_actions)):
                    delta = q_[:, t]
                    lastgaelam = delta + lam * lastgaelam
                    r[:, t] = lastgaelam

            token_level_score = torch.zeros_like(q)

            if self.config.step_granularity == "token":
                for i in range(micro_batch["input_ids"].shape[0]):
                    token_level_score[i, : max_positions[i] - 1] = r[i, : max_positions[i] - 1]
            # elif self.config.step_granularity == "whole":
            #     for i in range(micro_batch["input_ids"].shape[0]):
            #         token_level_score[i, max_positions[i] - 1] = r[i, : max_positions[i]]                
            elif self.config.step_granularity == "step":
                for i in range(micro_batch["input_ids"].shape[0]):
                    # Sum all valid rewards for this sample
                    total_reward = r[i, :max_positions[i]].sum()
                    # Assign to a single position (e.g., last valid position)
                    token_level_score[i, max_positions[i] - 1] = total_reward
            else:
                raise NotImplementedError

        return token_level_score, q

    def _optimizer_step(self):
        assert self.config.model.optim.grad_clip is not None

        if isinstance(self.reward_module, FSDP):
            grad_norm = self.reward_module.clip_grad_norm_(self.config.model.optim.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.reward_module.parameters(), max_norm=self.config.model.optim.grad_clip)
        self.reward_optimizer.step()
        return grad_norm

    # def istar_norm(self, token_level_scores):
    #     if self.config.istar_norm == "batch_norm":
    #         reverse_cumsum = torch.cumsum(token_level_scores.flip(dims=[1]), dim=-1).flip(dims=[1])
    #         token_level_scores = token_level_scores / (reverse_cumsum.abs().max() + 1e-6)
    #     return token_level_scores
    
    def istar_norm(self, token_level_scores):
        if self.config.istar_norm == "traj_norm":
            # token_level_scores: (B, T)
            reverse_cumsum = torch.cumsum(token_level_scores.flip(dims=[1]), dim=1).flip(dims=[1])  # (B, T)
            scale = reverse_cumsum.abs().amax(dim=1, keepdim=True) + 1e-6  # (B, 1)
            token_level_scores = token_level_scores / scale
        elif self.config.istar_norm == "batch_norm":
            reverse_cumsum = torch.cumsum(token_level_scores.flip(dims=[1]), dim=1).flip(dims=[1])  # (B, T)
            token_level_scores = token_level_scores / (reverse_cumsum.abs().max() + 1e-6)  # scalar
        return token_level_scores

    def compute_rm_score(self, data: DataProto):
        self.reward_module.eval()
        self.ref_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "acc", "old_log_probs"]
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        prompt_length = data.batch["input_ids"].shape[-1] - data.batch["responses"].shape[-1]

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        rm_scores_lst = []
        q_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                rm_score, q = self._forward_micro_batch(micro_batch, prompt_length)
            rm_scores_lst.append(rm_score)
            q_lst.append(q)
        rm_scores = torch.concat(rm_scores_lst, dim=0)
        q = torch.concat(q_lst, dim=0)

        rm_scores = self.istar_norm(rm_scores) # (bs, response_length)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == rm_scores.size(0), f"{len(indices)} vs. {rm_scores.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            rm_scores = rm_scores[revert_indices]

        return (
            rm_scores,
            q.detach(),
            {
                "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
                "reward_model/raw_reward": q.sum(dim=-1).mean().item(),
            },
        )


    # def update_rm(self, data: DataProto):
    #     self.reward_module.train()
    #     metrics = {}

    #     beta = self.config.model.get("beta_train", 0.05)

    #     select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "acc", "prompts", "old_log_probs"]

    #     # --- NEW: ensure uid exists in tensor batch ---
    #     if "uid" not in data.batch.keys():
    #         # uid originally lives in non_tensor_batch
    #         import numpy as np
    #         uid0 = np.asarray(data.non_tensor_batch["uid"])
    #         # must be numeric; convert object->int64 safely
    #         uid0 = uid0.astype(np.int64)
    #         data.batch["uid"] = torch.from_numpy(uid0).long()
    #     # --- END NEW ---

    #     # make sure it gets selected
    #     select_keys.append("uid")

    #     for key in ["Q_bc", "acc_bc"]:
    #         if key in data.batch.keys():
    #             select_keys.append(key)

    #     batch = data.select(batch_keys=select_keys).batch
    #     # # --- BEGIN: group-preserving batching for BT loss ---
    #     # cands_per_prompt = 8  # = n_agent * n_roll in your config
    #     # mbsz = self.config.micro_batch_size_per_gpu

    #     # assert mbsz == cands_per_prompt, (
    #     #     f"Expected micro_batch_size_per_gpu == cands_per_prompt, got {mbsz} vs {cands_per_prompt}"
    #     # )

    #     # # 1) sort the full tensordict by uid so each uid's 8 candidates are contiguous
    #     # uid = batch["uid"]
    #     # perm = torch.argsort(uid)
    #     # batch = batch[perm]

    #     # # 2) sanity: ensure every consecutive block of 8 has the same uid
    #     # N = batch["uid"].shape[0]
    #     # assert N % cands_per_prompt == 0, f"N={N} not divisible by cands_per_prompt={cands_per_prompt}"
    #     # uid_block = batch["uid"].view(-1, cands_per_prompt)
    #     # assert torch.all(uid_block.eq(uid_block[:, :1])), "UID blocks are not contiguous after sorting"

    #     # # 3) Build mini-batches in units of uid-blocks (never split a uid-block)
    #     # mini_bsz = self.config.mini_batch_size
    #     # assert mini_bsz % cands_per_prompt == 0, (
    #     #     f"mini_batch_size must be multiple of {cands_per_prompt} to preserve uid blocks, got {mini_bsz}"
    #     # )
    #     # blocks_per_mini = mini_bsz // cands_per_prompt
    #     # num_blocks = N // cands_per_prompt

    #     # dataloader = []
    #     # for b0 in range(0, num_blocks, blocks_per_mini):
    #     #     b1 = min(b0 + blocks_per_mini, num_blocks)
    #     #     i0 = b0 * cands_per_prompt
    #     #     i1 = b1 * cands_per_prompt
    #     #     dataloader.append(batch[i0:i1])
    #     # # --- END: group-preserving batching for BT loss ---

    #     dataloader = batch.split(self.config.mini_batch_size)

    #     rm_scores_lst = []
    #     q_lst = []

    #     total_loss_sum = 0.0
    #     total_count = 0
    #     total_pairs = 0

    #     for batch_idx, mini in enumerate(dataloader):
    #         if self.config.use_dynamic_bsz:
    #             max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
    #             micro_batches, _ = rearrange_micro_batches(batch=mini, max_token_len=max_token_len)
    #         else:
    #             micro_batches = mini.split(self.config.micro_batch_size_per_gpu)
    #             self.gradient_accumulation = self.config.mini_batch_size // self.config.micro_batch_size_per_gpu

    #         self.reward_optimizer.zero_grad()

    #         for mb in micro_batches:
    #             mb = mb.cuda()
    #             attention_mask = mb["attention_mask"]
    #             acc = mb["acc"]

    #             prompt_length = mb["prompts"].shape[-1]
    #             response_mask = attention_mask[:, prompt_length:].float()

    #             rm_score, q = self._forward_micro_batch(mb, prompt_length)

    #             rm_scores_lst.append(rm_score)
    #             q_lst.append(q.detach())

    #             ##############################debug_start###################################
    #             # with torch.no_grad():
    #             #     u, counts = torch.unique(mb["uid"], return_counts=True)
    #             #     maxK = int(counts.max().item()) if counts.numel() else 0
    #             #     both = 0
    #             #     for uu in u:
    #             #         idx = (mb["uid"] == uu).nonzero(as_tuple=False).squeeze(-1)
    #             #         a = mb["acc"][idx].float()
    #             #         if (a > 0.5).any() and (a <= 0.5).any():
    #             #             both += 1
    #             #     print("[BT-DEBUG][update_rm] mb_size", mb["uid"].numel(),
    #             #         "unique_uids", u.numel(),
    #             #         "max_uid_multiplicity", maxK,
    #             #         "uids_with_both_labels", both)
    #             # input()
    #             ##############################debug_end###################################

    #             if self.config.model.loss_type == "ce":
    #                 dpo_loss = compute_ce_dpo_loss_rm(q, acc, response_mask=response_mask, beta=beta)

    #             elif self.config.model.loss_type == "dpo":
    #                 dpo_loss, num_pairs = compute_bt_loss_rm(
    #                     token_level_scores=q,
    #                     acc=acc,
    #                     uid=mb["uid"],                  # <-- now tensor uid exists
    #                     response_mask=response_mask,
    #                     beta=beta,
    #                 )
    #                 total_pairs += num_pairs
                     

    #             elif self.config.model.loss_type == "ce+dpo":
    #                 dpo_loss_1 = compute_ce_dpo_loss_rm(q, acc, response_mask=response_mask, beta=beta)
    #                 dpo_loss_2, num_pairs = compute_bt_loss_rm(
    #                     token_level_scores=q,
    #                     acc=acc,
    #                     uid=mb["uid"],                  # <-- now tensor uid exists
    #                     response_mask=response_mask,
    #                     beta=beta,
    #                 )
    #                 dpo_loss = 0.5*dpo_loss_1+0.5*dpo_loss_2
    #                 total_pairs += num_pairs

    #             elif self.config.model.loss_type == "irl":
    #                 dpo_loss, num_pairs = compute_irl_loss_rm(
    #                     token_level_scores=q,
    #                     acc=acc,
    #                     uid=mb["uid"],                  # <-- now tensor uid exists
    #                     response_mask=response_mask,
    #                     beta=beta,
    #                 )
    #                 total_pairs += num_pairs

    #             elif self.config.model.loss_type == "irl+dpo":
    #                 dpo_loss_1, num_pairs = compute_irl_loss_rm(
    #                     token_level_scores=q,
    #                     acc=acc,
    #                     uid=mb["uid"],                  # <-- now tensor uid exists
    #                     response_mask=response_mask,
    #                     beta=beta,
    #                 )
    #                 total_pairs += num_pairs

    #                 dpo_loss_2, num_pairs = compute_bt_loss_rm(
    #                     token_level_scores=q,
    #                     acc=acc,
    #                     uid=mb["uid"],                  # <-- now tensor uid exists
    #                     response_mask=response_mask,
    #                     beta=beta,
    #                 )
    #                 dpo_loss = 0.5*dpo_loss_1+0.5*dpo_loss_2

    #             elif self.config.model.loss_type == "composite":
    #                 dpo_loss, additional_metric = composite_rm_loss(
    #                     token_level_scores=q,
    #                     acc=acc,
    #                     uid=mb["uid"],
    #                     response_mask=response_mask,
    #                     beta=beta,
    #                 )
    #                 metrics.update(additional_metric)
                
    #             elif self.config.model.loss_type == "bon_acc":
    #                 dpo_loss = compute_detach_dpo_loss_rm(
    #                     q, acc, Q_bc=mb["Q_bc"], acc_bc=mb["acc_bc"],
    #                     response_mask=response_mask, beta=beta, bon_mode="bon_acc",
    #                 )

    #             elif self.config.model.loss_type == "bon_rm":
    #                 dpo_loss = compute_detach_dpo_loss_rm(
    #                     q, acc, Q_bc=mb["Q_bc"], acc_bc=mb["acc_bc"],
    #                     response_mask=response_mask, beta=beta, bon_mode="bon_rm",
    #                 )
    #             else:
    #                 raise NotImplementedError

    #             append_to_dict(metrics, {"reward_model/dpo_loss": float(dpo_loss.detach().item())})

    #             mbsz = attention_mask.shape[0]
    #             total_loss_sum += float(dpo_loss.detach().item()) * mbsz
    #             total_count += mbsz

    #             if self.config.use_dynamic_bsz:
    #                 mbsz = attention_mask.shape[0]
    #                 loss = dpo_loss * (mbsz / self.config.ppo_mini_batch_size)
    #             else:
    #                 loss = dpo_loss / self.gradient_accumulation

    #             loss.backward()

    #         grad_norm = self._optimizer_step()
    #         append_to_dict(metrics, {"reward_model/grad_norm": float(grad_norm.detach().item())})

    #     self.reward_optimizer.zero_grad()

    #     rm_scores = torch.cat(rm_scores_lst, dim=0)
    #     q_all = torch.cat(q_lst, dim=0)

    #     rm_scores = self.istar_norm(rm_scores)
    #     metrics.update({
    #         "reward_model/num_pairs": total_pairs,
    #         "reward_model/loss": total_loss_sum / max(total_count, 1),
    #         "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
    #         "reward_model/raw_reward": q_all.sum(dim=-1).mean().item(),
    #     })

    #     return rm_scores, metrics

    def update_rm(self, data: DataProto):
        self.reward_module.train()
        metrics = {}

        beta = self.config.model.get("beta_train", 0.05)

        # keep old_log_probs "just in case"
        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "acc", "prompts", "old_log_probs"]

        # --- ensure uid exists in tensor batch ---
        if "uid" not in data.batch.keys():
            import numpy as np
            uid0 = np.asarray(data.non_tensor_batch["uid"]).astype(np.int64)
            data.batch["uid"] = torch.from_numpy(uid0).long()
        # --- end ensure uid ---

        select_keys.append("uid")

        for key in ["Q_bc", "acc_bc"]:
            if key in data.batch.keys():
                select_keys.append(key)

        batch = data.select(batch_keys=select_keys).batch
        dataloader = batch.split(self.config.mini_batch_size)

        rm_scores_lst = []
        q_lst = []

        total_loss_sum = 0.0
        total_count = 0
        total_pairs = 0

        for batch_idx, mini in enumerate(dataloader):
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini, max_token_len=max_token_len)
            else:
                micro_batches = mini.split(self.config.micro_batch_size_per_gpu)

                # <<< CHANGED: use actual number of micro-batches (replay may be smaller than config.mini_batch_size)
                self.gradient_accumulation = max(1, len(micro_batches))

            self.reward_optimizer.zero_grad()

            for mb in micro_batches:
                mb = mb.cuda()
                attention_mask = mb["attention_mask"]
                acc = mb["acc"]

                prompt_length = mb["prompts"].shape[-1]
                response_mask = attention_mask[:, prompt_length:].float()

                rm_score, q = self._forward_micro_batch(mb, prompt_length)

                rm_scores_lst.append(rm_score)
                q_lst.append(q.detach())

                if self.config.model.loss_type == "ce":
                    dpo_loss = compute_ce_dpo_loss_rm(q, acc, response_mask=response_mask, beta=beta)
                elif self.config.model.loss_type == "ce_full":
                    dpo_loss = compute_ce_hyper_loss_rm(q, acc, response_mask=response_mask, beta=beta, m=self.config.micro_batch_size_per_gpu,uid=mb["uid"],lambda_indirect=self.config.lambda_indirect_ce,)

                elif self.config.model.loss_type == "dpo":
                    dpo_loss, num_pairs = compute_bt_loss_rm(
                        token_level_scores=q,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )
                    total_pairs += num_pairs
                
                elif self.config.model.loss_type == "dpo_full":
                    dpo_loss, num_pairs = compute_bt_hyper_loss_rm(
                        token_level_scores=q,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                        lambda_indirect=self.config.lambda_indirect_dpo,
                    )
                    total_pairs += num_pairs 

                elif self.config.model.loss_type == "ce+dpo_full":
                    ce_loss = compute_ce_hyper_loss_rm(
                        token_level_scores=q,
                        acc=acc,
                        response_mask=response_mask,
                        beta=beta,
                        m=self.config.micro_batch_size_per_gpu,
                        lambda_indirect=self.config.lambda_indirect_ce,
                    )

                    bt_loss, num_valid_uid_groups = compute_bt_hyper_loss_rm(
                        token_level_scores=q,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                        lambda_indirect=self.config.lambda_indirect_dpo,
                    )

                    dpo_loss = ce_loss + self.config.lambda_dpo * bt_loss
                    total_pairs += num_valid_uid_groups 

                

                elif self.config.model.loss_type == "ce+dpo":
                    dpo_loss_1 = compute_ce_dpo_loss_rm(q, acc, response_mask=response_mask, beta=beta)
                    dpo_loss_2, num_pairs = compute_bt_loss_rm(
                        token_level_scores=q,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )
                    dpo_loss = dpo_loss_1 + self.config.lambda_dpo * dpo_loss_2
                    total_pairs += num_pairs

                elif self.config.model.loss_type == "irl":
                    dpo_loss, num_pairs = compute_irl_loss_rm(
                        token_level_scores=q,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )
                    total_pairs += num_pairs

                elif self.config.model.loss_type == "irl+dpo":
                    dpo_loss_1, num_pairs = compute_irl_loss_rm(
                        token_level_scores=q,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )
                    total_pairs += num_pairs

                    dpo_loss_2, num_pairs = compute_bt_loss_rm(
                        token_level_scores=q,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )
                    dpo_loss = 0.5 * dpo_loss_1 + 0.5 * dpo_loss_2

                elif self.config.model.loss_type == "composite":
                    dpo_loss, additional_metric = composite_rm_loss(
                        token_level_scores=q,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )
                    metrics.update(additional_metric)

                elif self.config.model.loss_type == "bon_acc":
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q, acc, Q_bc=mb["Q_bc"], acc_bc=mb["acc_bc"],
                        response_mask=response_mask, beta=beta, bon_mode="bon_acc",
                    )

                elif self.config.model.loss_type == "bon_rm":
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q, acc, Q_bc=mb["Q_bc"], acc_bc=mb["acc_bc"],
                        response_mask=response_mask, beta=beta, bon_mode="bon_rm",
                    )
                else:
                    raise NotImplementedError

                append_to_dict(metrics, {"reward_model/dpo_loss": float(dpo_loss.detach().item())})

                mbsz = attention_mask.shape[0]
                total_loss_sum += float(dpo_loss.detach().item()) * mbsz
                total_count += mbsz

                if self.config.use_dynamic_bsz:
                    loss = dpo_loss * (mbsz / self.config.ppo_mini_batch_size)
                else:
                    # uses the corrected self.gradient_accumulation above
                    loss = dpo_loss / self.gradient_accumulation

                # print_param_update(self.reward_module, "before backward")
                loss.backward()
            # print_param_update(self.reward_module, "after backward")
            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {"reward_model/grad_norm": float(grad_norm.detach().item())})
        # print_param_update(self.reward_module, "after step")
        # input()
        self.reward_optimizer.zero_grad()

        # <<< CHANGED: handle empty replay (no batches processed) safely
        if len(rm_scores_lst) == 0:
            # Return empty tensors + metrics; caller may skip using rm_scores.
            empty_scores = torch.zeros((0, 1), device=torch.device("cpu"))
            metrics.update({
                "reward_model/num_pairs": 0,
                "reward_model/loss": 0.0,
                "reward_model/reward": 0.0,
                "reward_model/raw_reward": 0.0,
            })
            return empty_scores, metrics

        rm_scores = torch.cat(rm_scores_lst, dim=0)
        q_all = torch.cat(q_lst, dim=0)

        rm_scores = self.istar_norm(rm_scores)
        metrics.update({
            "reward_model/num_pairs": total_pairs,
            "reward_model/loss": total_loss_sum / max(total_count, 1),
            "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
            "reward_model/raw_reward": q_all.sum(dim=-1).mean().item(),
        })

        return rm_scores, metrics


    def update_rm_eto(self, data: DataProto):
        """
        Update reward model using trajectory-level DPO training.
        This ensures preference pairs share the same initial context.
        """
        # make sure we are in training mode
        self.reward_module.train()
        metrics = {}

        beta = self.config.model.get("beta_train", 0.05)
        

        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "acc", "prompts", "old_log_probs"]
        batch = data.select(batch_keys=select_keys).batch
        dataloader = batch.split(self.config.mini_batch_size)

        rm_scores_lst = []
        q_lst = []

        for batch_idx, mini_batch_data in enumerate(dataloader):
            # split batch into micro_batches
            mini_batch = mini_batch_data
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_gpu)
                self.gradient_accumulation = self.config.mini_batch_size // self.config.micro_batch_size_per_gpu

            for micro_batch_data in micro_batches:
                micro_batch_data = micro_batch_data.cuda()
               
                prompt_ids = micro_batch_data["prompts"]
                prompt_length = prompt_ids.shape[-1]
                attention_mask = micro_batch_data["attention_mask"]
                response_mask = attention_mask[:, prompt_length:]

                rm_score, q = self._forward_micro_batch(micro_batch_data, prompt_length)

                rm_scores_lst.append(rm_score)
                q_lst.append(q.detach())

        rm_scores = torch.cat(rm_scores_lst, dim=0)
        q = torch.concat(q_lst, dim=0)

        # Use trajectory-level DPO loss
        loss = compute_eto_loss_rm(
            data=data,
            token_level_scores=q,
            beta=beta,
        )

        loss_data = {"reward_model/eto_loss": loss.detach().item()}
        append_to_dict(metrics, loss_data)
        
        self.reward_optimizer.zero_grad()
        loss.backward()
        grad_norm = self._optimizer_step()
        grad_data = {"reward_model/grad_norm": grad_norm.detach().item()}
        append_to_dict(metrics, grad_data)

        rm_scores = self.istar_norm(rm_scores)

        metrics.update(
            {
                "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
                "reward_model/raw_reward": q.sum(dim=-1).mean().item(),
            }
        )

        return rm_scores, metrics
    
class DataParallelPotentialRewardModel:
    """
    Composite Reward Model:
      token_reward_total = token_reward_process + outcome_coef * outcome_reward injected to tokens.

    Also supports training RM using either:
      - process-only q (stable) or
      - composite q_total (exactly matches policy reward definition)
    """

    def __init__(
        self,
        config,
        reward_module: nn.Module,
        ref_module: Optional[nn.Module],
        reward_optimizer: optim.Optimizer,
    ):
        self.config = config
        self.reward_module = reward_module
        self.ref_module = ref_module
        self.reward_optimizer = reward_optimizer

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        self.use_fused_kernels = self.config.model.get("use_fused_kernels", False)
        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)

        # --- composite knobs ---
        # where outcome comes from: "acc" or a tensor key in batch like "outcome_reward"
        self.outcome_key = self.config.get("outcome_key", "acc")
        # outcome shaping coefficient
        self.outcome_coef = float(self.config.get("outcome_coef", 1.0))
        # "last" or "uniform"
        self.outcome_mode = str(self.config.get("outcome_mode", "last"))
        # if True: RM loss uses q_total; else uses q_process
        self.train_on_composite = bool(self.config.get("train_on_composite", True))

        # normalization
        self.istar_norm_mode = self.config.get("istar_norm", None)

        print(
            f"[CompositeRM] remove_padding={self.use_remove_padding} "
            f"fused={self.use_fused_kernels} sp={self.ulysses_sequence_parallel_size} "
            f"outcome_key={self.outcome_key} outcome_coef={self.outcome_coef} outcome_mode={self.outcome_mode} "
            f"train_on_composite={self.train_on_composite} istar_norm={self.istar_norm_mode}"
        )

    # -------------------------
    # utilities
    # -------------------------

    @staticmethod
    def _cast_like(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return x.to(device=like.device, dtype=like.dtype)

    def _optimizer_step(self):
        assert self.config.model.optim.grad_clip is not None
        if isinstance(self.reward_module, FSDP):
            grad_norm = self.reward_module.clip_grad_norm_(self.config.model.optim.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.reward_module.parameters(), max_norm=self.config.model.optim.grad_clip)
        self.reward_optimizer.step()
        return grad_norm

    def istar_norm(self, token_level_scores: torch.Tensor) -> torch.Tensor:
        """
        Same behavior as your latest version: traj_norm or batch_norm.
        token_level_scores: (B, T_actions)
        """
        mode = self.istar_norm_mode
        if mode is None:
            return token_level_scores

        if mode == "traj_norm":
            reverse_cumsum = torch.cumsum(token_level_scores.flip(dims=[1]), dim=1).flip(dims=[1])
            scale = reverse_cumsum.abs().amax(dim=1, keepdim=True) + 1e-6
            return token_level_scores / scale

        if mode == "batch_norm":
            reverse_cumsum = torch.cumsum(token_level_scores.flip(dims=[1]), dim=1).flip(dims=[1])
            return token_level_scores / (reverse_cumsum.abs().max() + 1e-6)

        return token_level_scores

    def _inject_outcome_to_tokens(
        self,
        token_scores: torch.Tensor,     # (B, T_actions)
        max_positions: torch.Tensor,    # (B,) valid action length per sample
        outcome: torch.Tensor,          # (B,) scalar
        coef: float,
        mode: str,
    ) -> torch.Tensor:
        """
        Adds coef*outcome into token_scores either at last valid token or uniformly over valid tokens.
        """
        if coef == 0.0:
            return token_scores

        token_scores = token_scores.clone()
        outcome = self._cast_like(outcome, token_scores)

        B, T = token_scores.shape
        if mode == "last":
            for i in range(B):
                L = int(max_positions[i].item())
                if L > 0:
                    token_scores[i, L - 1] += coef * outcome[i]
            return token_scores

        if mode == "uniform":
            for i in range(B):
                L = int(max_positions[i].item())
                if L > 0:
                    token_scores[i, :L] += (coef * outcome[i]) / float(L)
            return token_scores

        raise ValueError(f"Unknown outcome_mode={mode}")

    def _get_outcome(self, micro_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Outcome scalar per trajectory. Default uses "acc".
        If you have external outcome reward, put it into batch under config.outcome_key.
        """
        if self.outcome_key not in micro_batch:
            raise KeyError(
                f"Outcome key '{self.outcome_key}' not found in batch. "
                f"Available keys: {list(micro_batch.keys())}"
            )
        out = micro_batch[self.outcome_key]
        # allow shape (B,1) or (B,)
        if out.dim() == 2 and out.size(1) == 1:
            out = out.squeeze(1)
        return out.float()

    # -------------------------
    # core: forward micro-batch
    # -------------------------

    def _forward_micro_batch(
        self,
        micro_batch: Dict[str, torch.Tensor],
        prompt_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          token_reward_total: (B, T_actions)  composite (process + outcome)
          token_reward_process: (B, T_actions)
          q_process: (B, T_actions)          rm_logprob - ref_logprob (trimmed)
          q_total_for_training: (B, T_actions) if train_on_composite else q_process
        """
        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]

        B, seqlen = input_ids.shape
        num_actions = seqlen - prompt_length

        # valid action lengths per sample
        max_positions = attention_mask[:, prompt_length:].sum(-1)  # (B,)

        # ---- compute rm_log_labels ----
        if self.use_remove_padding:
            # (you already have these functions)
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            position_ids_rmpad = index_first_axis(
                rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                indices
            ).transpose(0, 1)

            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)

            if self.ulysses_sequence_parallel_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.ulysses_sequence_parallel_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

            output = self.reward_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                use_cache=False,
            )

            if self.use_fused_kernels:
                rm_log_labels = output.log_probs.squeeze(0).to(torch.float32)
            else:
                rm_output_logits = output.logits.squeeze(0)
                rm_log_labels = verl_F.logprobs_from_logits(logits=rm_output_logits, labels=input_ids_rmpad_rolled)

            if self.ulysses_sequence_parallel_size > 1:
                rm_log_labels = gather_outpus_and_unpad(rm_log_labels, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            rm_log_labels = pad_input(
                hidden_states=rm_log_labels.unsqueeze(-1),
                indices=indices,
                batch=B,
                seqlen=seqlen
            ).squeeze(-1)[:, -num_actions - 1 : -1]  # (B, num_actions)

        else:
            output = self.reward_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )

            if self.use_fused_kernels:
                rm_log_labels = output.log_probs[:, :-1].to(torch.float32)[:, -num_actions:]  # (B, num_actions)
            else:
                rm_output_logits = output.logits
                rm_log_labels_full = verl_F.logprobs_from_logits(
                    logits=rm_output_logits[:, :-1, :],
                    labels=input_ids[:, 1:]
                )
                rm_log_labels = rm_log_labels_full[:, -num_actions:]  # (B, num_actions)

        # ---- compute ref_log_labels ----
        if self.ref_module is not None:
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                if self.use_remove_padding and self.ulysses_sequence_parallel_size > 1:
                    ref_output = self.ref_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        use_cache=False,
                    )
                    if self.use_fused_kernels:
                        ref_log_labels = ref_output.log_probs.squeeze(0).to(torch.float32)
                    else:
                        ref_output_logits = ref_output.logits.squeeze(0)
                        ref_log_labels = verl_F.logprobs_from_logits(logits=ref_output_logits, labels=input_ids_rmpad_rolled)

                    ref_log_labels = gather_outpus_and_unpad(ref_log_labels, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    ref_log_labels = pad_input(ref_log_labels.unsqueeze(-1), indices=indices, batch=B, seqlen=seqlen).squeeze(-1)[:, -num_actions - 1 : -1]
                else:
                    ref_output = self.ref_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        use_cache=False,
                    )
                    if self.use_fused_kernels:
                        ref_log_labels_full = ref_output.log_probs[:, :-1].to(torch.float32)
                    else:
                        ref_output_logits = ref_output.logits
                        ref_log_labels_full = verl_F.logprobs_from_logits(
                            logits=ref_output_logits[:, :-1, :],
                            labels=input_ids[:, 1:]
                        )
                    ref_log_labels = ref_log_labels_full[:, -num_actions:]
        else:
            # uses old log probs if no ref module
            ref_log_labels = micro_batch["old_log_probs"][:, -num_actions:]

        # IMPORTANT: actually cast
        ref_log_labels = ref_log_labels.to(dtype=rm_log_labels.dtype, device=rm_log_labels.device)

        # q_process: (B, num_actions)
        q_process = rm_log_labels[:, -num_actions:] - ref_log_labels[:, -num_actions:]

        # trim invalid action tokens
        for i in range(B):
            L = int(max_positions[i].item())
            if L < num_actions:
                q_process[i, L:] = 0

        # ---- build process token reward (no grad needed) ----
        with torch.no_grad():
            lam = float(self.config.get("lambda", 0.0))
            beta = float(self.config.model.get("beta_train", 0.05))

            if lam == 0.0:
                r_process = q_process * beta
            else:
                acc = micro_batch["acc"].float().to(q_process.device)
                q_ = q_process * beta
                r_process = torch.zeros_like(q_process)
                lastgaelam = 0

                # optional PRIME-style constraint to force last token to match outcome
                for i in range(B):
                    L = int(max_positions[i].item())
                    if L <= 0:
                        continue
                    if getattr(self.config, "prime_use_gt", False):
                        q_[i, L - 1] = acc[i] - q_[i, : L - 1].sum()
                    if L < num_actions:
                        q_[i, L:] = 0

                for t in reversed(range(num_actions)):
                    delta = q_[:, t]
                    lastgaelam = delta + lam * lastgaelam
                    r_process[:, t] = lastgaelam

            token_reward_process = torch.zeros_like(q_process)

            if self.config.step_granularity == "token":
                for i in range(B):
                    L = int(max_positions[i].item())
                    if L > 0:
                        token_reward_process[i, : L] = r_process[i, : L]
            elif self.config.step_granularity == "step":
                for i in range(B):
                    L = int(max_positions[i].item())
                    if L > 0:
                        token_reward_process[i, L - 1] = r_process[i, :L].sum()
            else:
                raise NotImplementedError(f"Unknown step_granularity={self.config.step_granularity}")

        # ---- outcome injection (composite reward for policy) ----
        outcome = self._get_outcome(micro_batch)  # (B,)
        token_reward_total = self._inject_outcome_to_tokens(
            token_scores=token_reward_process,
            max_positions=max_positions,
            outcome=outcome,
            coef=self.outcome_coef,
            mode=self.outcome_mode,
        )

        # ---- build q_total for training (optional) ----
        if self.train_on_composite:
            q_total = self._inject_outcome_to_tokens(
                token_scores=q_process,
                max_positions=max_positions,
                outcome=outcome,
                coef=self.outcome_coef,
                mode=self.outcome_mode,
            )
        else:
            q_total = q_process

        return token_reward_total, token_reward_process, q_process, q_total

    # -------------------------
    # public: compute score (for PPO)
    # -------------------------

    def compute_rm_score(self, data):
        self.reward_module.eval()
        if self.ref_module is not None:
            self.ref_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        select_keys = [
            "responses", "input_ids", "attention_mask", "position_ids",
            "acc", "old_log_probs",
        ]
        # outcome key might be different from acc
        if self.outcome_key not in select_keys:
            select_keys.append(self.outcome_key)

        batch = data.select(batch_keys=select_keys).batch
        prompt_length = data.batch["input_ids"].shape[-1] - data.batch["responses"].shape[-1]

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        total_lst = []
        process_lst = []
        q_lst = []

        for micro_batch in micro_batches:
            with torch.no_grad():
                token_total, token_process, q_process, _q_total = self._forward_micro_batch(micro_batch, prompt_length)
            total_lst.append(token_total)
            process_lst.append(token_process)
            q_lst.append(q_process)

        token_total = torch.cat(total_lst, dim=0)
        token_process = torch.cat(process_lst, dim=0)
        q_process = torch.cat(q_lst, dim=0)

        # normalize composite reward (this matches what PPO consumes)
        token_total = self.istar_norm(token_total)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            token_total = token_total[revert_indices]
            token_process = token_process[revert_indices]
            q_process = q_process[revert_indices]

        # compute outcome-only contribution for logging
        with torch.no_grad():
            attn = batch["attention_mask"]
            mp = attn[:, prompt_length:].sum(-1)
            outcome = batch[self.outcome_key].float()
            outcome_tokens = torch.zeros_like(token_total)
            outcome_tokens = self._inject_outcome_to_tokens(outcome_tokens, mp, outcome, self.outcome_coef, self.outcome_mode)

        metrics = {
            "reward_model/total_reward": token_total.sum(dim=-1).mean().item(),
            "reward_model/process_reward": token_process.sum(dim=-1).mean().item(),
            "reward_model/outcome_reward": outcome_tokens.sum(dim=-1).mean().item(),
            "reward_model/raw_q_process": q_process.sum(dim=-1).mean().item(),
        }
        return token_total, q_process.detach(), metrics

    # -------------------------
    # public: update RM
    # -------------------------

    def update_rm(self, data):
        self.reward_module.train()
        metrics: Dict[str, float] = {}

        beta = float(self.config.model.get("beta_train", 0.05))

        select_keys = [
            "input_ids", "responses", "attention_mask", "position_ids",
            "acc", "prompts", "old_log_probs"
        ]
        if self.outcome_key not in select_keys:
            select_keys.append(self.outcome_key)

        # ensure uid exists in tensor batch (same as your code)
        if "uid" not in data.batch.keys():
            import numpy as np
            uid0 = np.asarray(data.non_tensor_batch["uid"]).astype(np.int64)
            data.batch["uid"] = torch.from_numpy(uid0).long()
        select_keys.append("uid")

        for key in ["Q_bc", "acc_bc"]:
            if key in data.batch.keys():
                select_keys.append(key)

        batch = data.select(batch_keys=select_keys).batch
        dataloader = batch.split(self.config.mini_batch_size)

        total_reward_lst = []
        process_reward_lst = []
        q_process_lst = []

        total_loss_sum = 0.0
        total_count = 0
        total_pairs = 0

        for batch_idx, mini in enumerate(dataloader):
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini, max_token_len=max_token_len)
                grad_denom = float(self.config.ppo_mini_batch_size)
            else:
                micro_batches = mini.split(self.config.micro_batch_size_per_gpu)
                self.gradient_accumulation = self.config.mini_batch_size // self.config.micro_batch_size_per_gpu
                grad_denom = float(self.gradient_accumulation)

            self.reward_optimizer.zero_grad()

            for mb in micro_batches:
                mb = mb.cuda()
                attention_mask = mb["attention_mask"]
                acc = mb["acc"]
                prompt_length = mb["prompts"].shape[-1]
                response_mask = attention_mask[:, prompt_length:].float()

                token_total, token_process, q_process, q_for_loss = self._forward_micro_batch(mb, prompt_length)

                total_reward_lst.append(token_total.detach())
                process_reward_lst.append(token_process.detach())
                q_process_lst.append(q_process.detach())

                # ---- choose loss input: composite or process-only ----
                # q_for_loss is already selected based on self.train_on_composite
                if self.config.model.loss_type == "ce":
                    dpo_loss = compute_ce_dpo_loss_rm(q_for_loss, acc, response_mask=response_mask, beta=beta)
                
                elif self.config.model.loss_type == "ce_full":
                    dpo_loss = compute_ce_hyper_loss_rm(q_for_loss, acc, response_mask=response_mask, beta=beta, m=self.config.micro_batch_size_per_gpu,uid=mb["uid"],lambda_indirect=self.config.lambda_indirect_ce,)

                elif self.config.model.loss_type == "dpo":
                    dpo_loss, num_pairs = compute_bt_loss_rm(
                        token_level_scores=q_for_loss,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )
                    total_pairs += num_pairs
                
                elif self.config.model.loss_type == "dpo_full":
                    dpo_loss, num_pairs = compute_bt_hyper_loss_rm(
                        token_level_scores=q_for_loss,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                        lambda_indirect=self.config.lambda_indirect_dpo,
                    )
                    total_pairs += num_pairs 

                elif self.config.model.loss_type == "ce+dpo_full":
                    ce_loss = compute_ce_hyper_loss_rm(
                        token_level_scores=q_for_loss,
                        acc=acc,
                        response_mask=response_mask,
                        beta=beta,
                        m=self.config.micro_batch_size_per_gpu,
                        lambda_indirect=self.config.lambda_indirect_ce,
                    )

                    # print("================================")
                    # print("ce loss completed")
                    # print("================================")

                    bt_loss, num_valid_uid_groups = compute_bt_hyper_loss_rm(
                        token_level_scores=q_for_loss,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                        lambda_indirect=self.config.lambda_indirect_dpo,
                    )

                    # print("================================")
                    # print("dpo loss completed")
                    # print("================================")

                    dpo_loss = ce_loss + self.config.lambda_dpo * bt_loss
                    total_pairs += num_valid_uid_groups 

                    # ---------- debug scale ----------
                    if getattr(self.config, "debug_lambda_dpo", False):
                        ce_q_norm, ce_q_abs_mean, ce_q_abs_max = tensor_grad_norm_from_loss(
                            ce_loss, q_for_loss, retain_graph=True
                        )

                        bt_q_norm, bt_q_abs_mean, bt_q_abs_max = tensor_grad_norm_from_loss(
                            bt_loss, q_for_loss, retain_graph=True
                        )

                        grad_ratio = bt_q_norm / max(ce_q_norm, 1e-12)
                        suggested_lambda = 1.0 / max(grad_ratio, 1e-12)

                        print("\n========== LAMBDA_DPO DEBUG CHEAP ==========")
                        print(f"ce_loss:                   {ce_loss.detach().item():.8f}")
                        print(f"bt_loss:                   {bt_loss.detach().item():.8f}")
                        print(f"lambda_dpo:                {self.config.lambda_dpo:.8g}")
                        print(f"lambda_dpo * bt_loss:      {(self.config.lambda_dpo * bt_loss).detach().item():.8f}")

                        print("\n--- Gradient wrt q_for_loss ---")
                        print(f"CE dq norm:                {ce_q_norm:.8f}")
                        print(f"BT dq norm:                {bt_q_norm:.8f}")
                        print(f"lambda * BT dq norm:       {self.config.lambda_dpo * bt_q_norm:.8f}")
                        print(f"BT / CE dq ratio:          {grad_ratio:.8f}")
                        print(f"suggest lambda_dpo:        {suggested_lambda:.8g}")

                        print("\n--- dq abs stats ---")
                        print(f"CE dq abs mean/max:        {ce_q_abs_mean:.8e} / {ce_q_abs_max:.8e}")
                        print(f"BT dq abs mean/max:        {bt_q_abs_mean:.8e} / {bt_q_abs_max:.8e}")
                        print("===========================================\n")


                elif self.config.model.loss_type == "ce+dpo":
                    l1 = compute_ce_dpo_loss_rm(q_for_loss, acc, response_mask=response_mask, beta=beta)
                    l2, num_pairs = compute_bt_loss_rm(
                        token_level_scores=q_for_loss,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )
                    dpo_loss = l1 + 0.1 * l2
                    total_pairs += num_pairs

                elif self.config.model.loss_type == "irl":
                    dpo_loss, num_pairs = compute_irl_loss_rm(
                        token_level_scores=q_for_loss,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )
                    total_pairs += num_pairs

                elif self.config.model.loss_type == "irl+dpo":
                    l1, np1 = compute_irl_loss_rm(
                        token_level_scores=q_for_loss,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )
                    l2, np2 = compute_bt_loss_rm(
                        token_level_scores=q_for_loss,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )
                    dpo_loss = 0.5 * l1 + 0.5 * l2
                    total_pairs += (np1 + np2)

                elif self.config.model.loss_type == "composite":
                    dpo_loss, additional_metric = composite_rm_loss(
                        token_level_scores=q_for_loss,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )
                    metrics.update(additional_metric)

                elif self.config.model.loss_type == "bon_acc":
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q_for_loss, acc, Q_bc=mb["Q_bc"], acc_bc=mb["acc_bc"],
                        response_mask=response_mask, beta=beta, bon_mode="bon_acc",
                    )

                elif self.config.model.loss_type == "bon_rm":
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q_for_loss, acc, Q_bc=mb["Q_bc"], acc_bc=mb["acc_bc"],
                        response_mask=response_mask, beta=beta, bon_mode="bon_rm",
                    )

                else:
                    raise NotImplementedError

                append_to_dict(metrics, {"reward_model/loss_micro": float(dpo_loss.detach().item())})

                mbsz = attention_mask.shape[0]
                total_loss_sum += float(dpo_loss.detach().item()) * mbsz
                total_count += mbsz

                if self.config.use_dynamic_bsz:
                    loss = dpo_loss * (mbsz / grad_denom)
                else:
                    loss = dpo_loss / grad_denom

               
                # print_param_update(self.reward_module, "before backward")
                loss.backward()
            # print_param_update(self.reward_module, "after backward")
            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {"reward_model/grad_norm": float(grad_norm.detach().item())})
        # print_param_update(self.reward_module, "after step")
        # input()
        self.reward_optimizer.zero_grad()


        # <<< CHANGED: handle empty replay (no batches processed) safely
        if len(total_reward_lst) == 0:
            # Return empty tensors + metrics; caller may skip using rm_scores.
            empty_scores = torch.zeros((0, 1), device=torch.device("cpu"))
            metrics.update({
                "reward_model/num_pairs": 0,
                "reward_model/loss": 0.0,
                "reward_model/reward": 0.0,
                "reward_model/raw_reward": 0.0,
            })
            return empty_scores, metrics

        token_total = torch.cat(total_reward_lst, dim=0)
        token_process = torch.cat(process_reward_lst, dim=0)
        q_process_all = torch.cat(q_process_lst, dim=0)

        token_total = self.istar_norm(token_total)

        # outcome-only part for logging (computed from batch tensor; ok if dynamic bsz was used)
        with torch.no_grad():
            prompt_length = batch["prompts"].shape[-1]
            mp = batch["attention_mask"][:, prompt_length:].sum(-1)
            outcome = batch[self.outcome_key].float()
            outcome_tokens = torch.zeros_like(token_total)
            outcome_tokens = self._inject_outcome_to_tokens(outcome_tokens, mp, outcome, self.outcome_coef, self.outcome_mode)

        metrics.update({
            "reward_model/num_pairs": int(total_pairs),
            "reward_model/loss": total_loss_sum / max(total_count, 1),
            "reward_model/total_reward": token_total.sum(dim=-1).mean().item(),
            "reward_model/process_reward": token_process.sum(dim=-1).mean().item(),
            "reward_model/outcome_reward": outcome_tokens.sum(dim=-1).mean().item(),
            "reward_model/raw_q_process": q_process_all.sum(dim=-1).mean().item(),
        })

        return token_total, metrics




def _extract_last_hidden_states(output) -> torch.Tensor:
    """
    Returns:
        h: (B, S, H) or (1, NNZ, H) depending on input packing.
    """
    # HF-style: output.hidden_states exists if output_hidden_states=True
    if hasattr(output, "hidden_states") and output.hidden_states is not None:
        return output.hidden_states[-1]
    # Some models may expose last_hidden_state
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state
    raise RuntimeError(
        "Reward backbone did not return hidden states. "
        "Ensure the forward call sets output_hidden_states=True and return_dict=True."
    )


class DataParallelLMHeadISTARRewardModel:
    """
    A reward class that COMPLIES with your existing construction code:
      - reward_module is already built + FSDP-wrapped in the worker
      - reward_optimizer is created as AdamW(reward_module.parameters(), ...)
      - we DO NOT change that construction code.

    To implement "LM + linear head" without touching construction, we:
      1) create `self.head = nn.Linear(hidden_size, 1)` inside this class
      2) ATTACH head params to `reward_optimizer` by adding a param_group
         (so optimizer updates head too, without changing worker code)
      3) compute token-level scores from backbone hidden states, then subtract
         reference scores (if ref_module exists) to form q.

    Public API matches your shown class:
      - compute_rm_score
      - update_rm
      - update_rm_eto
      - plus helper methods/properties.
    """

    def __init__(self, config, reward_module: nn.Module, ref_module: nn.Module, reward_optimizer: optim.Optimizer):
        self.config = config
        self.reward_module = reward_module
        self.ref_module = ref_module
        self.reward_optimizer = reward_optimizer

        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Reward model use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.model.get("use_fused_kernels", False)
        print(f"Reward model use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)

        # ---------- Build linear head (LM + linear head) ----------
        hidden_size = None
        # Prefer backbone config if available
        if hasattr(self.reward_module, "config"):
            cfg = self.reward_module.config
            hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or getattr(cfg, "d_model", None)

        # Fallbacks (in case reward_module.config is not accessible via FSDP wrapper)
        if hidden_size is None and hasattr(reward_module, "_fsdp_wrapped_module"):
            base = reward_module._fsdp_wrapped_module
            if hasattr(base, "config"):
                cfg = base.config
                hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or getattr(cfg, "d_model", None)

        if hidden_size is None:
            raise ValueError(
                "Cannot infer hidden_size for linear head. "
                "Ensure reward_module has a HF config with hidden_size/n_embd/d_model."
            )

        self.head = nn.Linear(hidden_size, 1, bias=True)

        # Put head on GPU (same device as reward module)
        # (reward_module is FSDP on current cuda device in your worker)
        self.head = self.head.cuda()
        # Keep head in fp32 by default for stability (you can change if you want)
        self.head = self.head.to(torch.bfloat16)

        # IMPORTANT: attach head params to the *existing* optimizer without changing construction code
        self._maybe_add_head_params_to_optimizer()

        # Optional: a reference head (frozen) to define q = head(h) - ref_head(h_ref)
        # If you want ref_head to track reward head initially, copy weights.
        self.ref_head = nn.Linear(hidden_size, 1, bias=True).cuda().to(torch.bfloat16)
        self.ref_head.load_state_dict(self.head.state_dict())
        for p in self.ref_head.parameters():
            p.requires_grad_(False)

    def _maybe_add_head_params_to_optimizer(self):
        """
        The worker constructs:
            reward_optimizer = AdamW(reward_module.parameters(), ...)
        So head params are missing. We add them here by inserting a new param_group.
        """
        try:
            opt_param_ids = {id(p) for g in self.reward_optimizer.param_groups for p in g["params"]}
            head_params = [p for p in self.head.parameters() if p.requires_grad]
            missing = [p for p in head_params if id(p) not in opt_param_ids]
            if missing:
                # Use same hyperparams as the first group (typical AdamW)
                base_group = self.reward_optimizer.param_groups[0]
                new_group = {
                    "params": missing,
                    "lr": base_group.get("lr", 1e-4),
                    "betas": base_group.get("betas", (0.9, 0.999)),
                    "eps": base_group.get("eps", 1e-8),
                    "weight_decay": base_group.get("weight_decay", 0.0),
                }
                self.reward_optimizer.add_param_group(new_group)
        except Exception as e:
            # If something odd about optimizer, fail loudly (better than silent no-training)
            raise RuntimeError(f"Failed to add head params to optimizer: {e}")

    def _compute_token_scores_from_backbone(
        self,
        backbone: nn.Module,
        head: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        use_remove_padding: bool,
        ulysses_sp_size: int,
    ) -> torch.Tensor:
        """
        Returns:
            scores_full: (B, S) float32 token-level scalar scores.
        """
        batch_size, seqlen = input_ids.shape

        if use_remove_padding:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (nnz,1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, nnz)

            position_ids_rmpad = index_first_axis(
                rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                indices
            ).transpose(0, 1)  # (1, nnz)

            if ulysses_sp_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=ulysses_sp_size
                )

            out = backbone(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            h = _extract_last_hidden_states(out)  # (1, nnz, H) (or (1, padded, H))
            h = h.squeeze(0)  # (nnz, H)

            scores_rmpad = head(h).squeeze(-1).to(torch.float32)  # (nnz,)

            if ulysses_sp_size > 1:
                scores_rmpad = gather_outpus_and_unpad(scores_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            scores_full = pad_input(
                hidden_states=scores_rmpad.unsqueeze(-1),
                indices=indices,
                batch=batch_size,
                seqlen=seqlen,
            ).squeeze(-1)  # (B, S)

            return scores_full

        # no remove-padding
        out = backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        h = _extract_last_hidden_states(out)  # (B, S, H)
        scores_full = head(h).squeeze(-1).to(torch.float32)  # (B, S)
        return scores_full

    def _forward_micro_batch(self, micro_batch, prompt_length):
        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]

        batch_size, seqlen = input_ids.shape
        num_actions = input_ids.shape[-1] - prompt_length
        max_positions = attention_mask[:, prompt_length:].sum(-1)

        # --- reward scores (with grad) ---
        rm_scores_full = self._compute_token_scores_from_backbone(
            backbone=self.reward_module,
            head=self.head,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_remove_padding=self.use_remove_padding,
            ulysses_sp_size=self.ulysses_sequence_parallel_size,
        )
        # align like your original: take action-region token positions using [-num_actions-1:-1]
        rm_scores_tok = rm_scores_full[:, -num_actions - 1 : -1]  # (B, T_actions)


        # q is the learned token-level score
        q = rm_scores_tok 

        # trim invalid tail
        for i in range(batch_size):
            q[i, max_positions[i] :] = 0

        # reward computation does not need gradient. only q needs
        with torch.no_grad():
            lam = self.config.get("lambda", 0.0)
            beta = self.config.model.get("beta_train", 0.05)

            if lam == 0.0:
                r = q * beta
            else:
                acc = micro_batch["acc"]
                q_ = q * beta
                r = torch.zeros_like(q)
                lastgaelam = 0

                for i in range(q.shape[0]):
                    if self.config.prime_use_gt:
                        q_[i, max_positions[i] - 1] = acc[i] - q_[i, : max_positions[i] - 1].sum()
                    q_[i, max_positions[i] :] = 0

                for t in reversed(range(num_actions)):
                    delta = q_[:, t]
                    lastgaelam = delta + lam * lastgaelam
                    r[:, t] = lastgaelam

            token_level_score = torch.zeros_like(q)

            if self.config.step_granularity == "token":
                for i in range(batch_size):
                    token_level_score[i, : max_positions[i] - 1] = r[i, : max_positions[i] - 1]
            elif self.config.step_granularity == "step":
                for i in range(batch_size):
                    token_level_score[i, max_positions[i] - 1] = r[i, : max_positions[i]].sum()
            else:
                raise NotImplementedError

        return token_level_score, q

    def _optimizer_step(self):
        assert self.config.model.optim.grad_clip is not None

        # Clip backbone grads
        if isinstance(self.reward_module, FSDP):
            grad_norm_backbone = self.reward_module.clip_grad_norm_(self.config.model.optim.grad_clip)
        else:
            grad_norm_backbone = torch.nn.utils.clip_grad_norm_(
                self.reward_module.parameters(), max_norm=self.config.model.optim.grad_clip
            )

        # Clip head grads too (head is not inside FSDP wrapper given your construction)
        grad_norm_head = torch.nn.utils.clip_grad_norm_(
            self.head.parameters(), max_norm=self.config.model.optim.grad_clip
        )

        self.reward_optimizer.step()

        # return a conservative combined norm (max of the two)
        try:
            gb = grad_norm_backbone if torch.is_tensor(grad_norm_backbone) else torch.tensor(float(grad_norm_backbone), device=grad_norm_head.device)
            gh = grad_norm_head if torch.is_tensor(grad_norm_head) else torch.tensor(float(grad_norm_head), device=grad_norm_head.device)
            return torch.maximum(gb, gh)
        except Exception:
            return grad_norm_backbone

    def istar_norm(self, token_level_scores):
        if self.config.istar_norm == "batch_norm":
            reverse_cumsum = torch.cumsum(token_level_scores.flip(dims=[1]), dim=-1).flip(dims=[1])
            token_level_scores = token_level_scores / (reverse_cumsum.abs().max() + 1e-6)
        return token_level_scores

    def compute_rm_score(self, data: DataProto):
        self.reward_module.eval()
        self.head.eval()
        if self.ref_module is not None:
            self.ref_module.eval()
        self.ref_head.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "acc", "old_log_probs"]
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]

        prompt_length = data.batch["input_ids"].shape[-1] - data.batch["responses"].shape[-1]

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        rm_scores_lst, q_lst = [], []
        for micro_batch in micro_batches:
            with torch.no_grad():
                rm_score, q = self._forward_micro_batch(micro_batch, prompt_length)
            rm_scores_lst.append(rm_score)
            q_lst.append(q)

        rm_scores = torch.concat(rm_scores_lst, dim=0)
        q = torch.concat(q_lst, dim=0)

        rm_scores = self.istar_norm(rm_scores)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == rm_scores.size(0), f"{len(indices)} vs. {rm_scores.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            rm_scores = rm_scores[revert_indices]
            q = q[revert_indices]

        return (
            rm_scores,
            q.detach(),
            {
                "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
                "reward_model/raw_reward": q.sum(dim=-1).mean().item(),
            },
        )

    def update_rm(self, data: DataProto):
        self.reward_module.train()
        self.head.train()
        metrics = {}

        beta = self.config.model.get("beta_train", 0.05)

        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "acc", "prompts", "old_log_probs"]

        # --- ensure uid exists in tensor batch (same as your patched version) ---
        if "uid" not in data.batch.keys():
            import numpy as np
            uid0 = np.asarray(data.non_tensor_batch["uid"]).astype(np.int64)
            data.batch["uid"] = torch.from_numpy(uid0).long()
        select_keys.append("uid")

        for key in ["Q_bc", "acc_bc"]:
            if key in data.batch.keys():
                select_keys.append(key)

        batch = data.select(batch_keys=select_keys).batch
        dataloader = batch.split(self.config.mini_batch_size)

        rm_scores_lst, q_lst = [], []

        for _, mini in enumerate(dataloader):
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini, max_token_len=max_token_len)
            else:
                micro_batches = mini.split(self.config.micro_batch_size_per_gpu)
                self.gradient_accumulation = self.config.mini_batch_size // self.config.micro_batch_size_per_gpu

            self.reward_optimizer.zero_grad()

            for mb in micro_batches:
                mb = mb.cuda()
                attention_mask = mb["attention_mask"]
                acc = mb["acc"]

                prompt_length = mb["prompts"].shape[-1]
                response_mask = attention_mask[:, prompt_length:].float()

                rm_score, q = self._forward_micro_batch(mb, prompt_length)

                rm_scores_lst.append(rm_score)
                q_lst.append(q.detach())

                if self.config.model.loss_type == "ce":
                    dpo_loss = compute_ce_dpo_loss_rm(q, acc, response_mask=response_mask, beta=beta)
                elif self.config.model.loss_type == "ce_full":
                    dpo_loss = compute_ce_hyper_loss_rm(q, acc, response_mask=response_mask, beta=beta, m=self.config.micro_batch_size_per_gpu,uid=mb["uid"],lambda_indirect=self.config.lambda_indirect_ce,)

                elif self.config.model.loss_type == "dpo":
                    dpo_loss, num_pairs = compute_bt_loss_rm(
                        token_level_scores=q,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )
                    total_pairs += num_pairs
                
                elif self.config.model.loss_type == "dpo_full":
                    dpo_loss, num_pairs = compute_bt_hyper_loss_rm(
                        token_level_scores=q,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                        lambda_indirect=self.config.lambda_indirect_dpo,
                    )
                    total_pairs += num_pairs 

                elif self.config.model.loss_type == "ce+dpo_full":
                    ce_loss = compute_ce_hyper_loss_rm(
                        token_level_scores=q,
                        acc=acc,
                        response_mask=response_mask,
                        beta=beta,
                        m=self.config.micro_batch_size_per_gpu,
                        lambda_indirect=self.config.lambda_indirect_ce,
                    )

                    bt_loss, num_valid_uid_groups = compute_bt_hyper_loss_rm(
                        token_level_scores=q,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                        lambda_indirect=self.config.lambda_indirect_dpo,
                    )

                    dpo_loss = ce_loss + self.config.lambda_dpo * bt_loss
                    total_pairs += num_valid_uid_groups 

                

                elif self.config.model.loss_type == "ce+dpo":
                    dpo_loss_1 = compute_ce_dpo_loss_rm(q, acc, response_mask=response_mask, beta=beta)
                    dpo_loss_2, num_pairs = compute_bt_loss_rm(
                        token_level_scores=q,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )
                    dpo_loss = dpo_loss_1 + self.config.lambda_dpo * dpo_loss_2
                    total_pairs += num_pairs

                elif self.config.model.loss_type == "irl":
                    dpo_loss, num_pairs = compute_irl_loss_rm(
                        token_level_scores=q,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )
                    total_pairs += num_pairs

                elif self.config.model.loss_type == "irl+dpo":
                    dpo_loss_1, num_pairs = compute_irl_loss_rm(
                        token_level_scores=q,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )
                    total_pairs += num_pairs

                    dpo_loss_2, num_pairs = compute_bt_loss_rm(
                        token_level_scores=q,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )
                    dpo_loss = 0.5 * dpo_loss_1 + 0.5 * dpo_loss_2

                elif self.config.model.loss_type == "composite":
                    dpo_loss, additional_metric = composite_rm_loss(
                        token_level_scores=q,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )
                    metrics.update(additional_metric)

                elif self.config.model.loss_type == "bon_acc":
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q,
                        acc,
                        Q_bc=mb["Q_bc"],
                        acc_bc=mb["acc_bc"],
                        response_mask=response_mask,
                        beta=beta,
                        bon_mode="bon_acc",
                    )

                elif self.config.model.loss_type == "bon_rm":
                    dpo_loss = compute_detach_dpo_loss_rm(
                        q,
                        acc,
                        Q_bc=mb["Q_bc"],
                        acc_bc=mb["acc_bc"],
                        response_mask=response_mask,
                        beta=beta,
                        bon_mode="bon_rm",
                    )
                else:
                    raise NotImplementedError

                append_to_dict(metrics, {"reward_model/dpo_loss": float(dpo_loss.detach().item())})

                if self.config.use_dynamic_bsz:
                    mbsz = attention_mask.shape[0]
                    loss = dpo_loss * (mbsz / self.config.ppo_mini_batch_size)
                else:
                    loss = dpo_loss / self.gradient_accumulation

                loss.backward()

            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {"reward_model/grad_norm": float(grad_norm.detach().item())})

        self.reward_optimizer.zero_grad()

        rm_scores = torch.cat(rm_scores_lst, dim=0)
        q_all = torch.cat(q_lst, dim=0)

        rm_scores = self.istar_norm(rm_scores)
        metrics.update(
            {
                "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
                "reward_model/raw_reward": q_all.sum(dim=-1).mean().item(),
            }
        )

        return rm_scores, metrics

    def update_rm_eto(self, data: DataProto):
        """
        Trajectory-level DPO update (ETO), same structure as your original.
        """
        self.reward_module.train()
        self.head.train()
        metrics = {}

        beta = self.config.model.get("beta_train", 0.05)

        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "acc", "prompts", "old_log_probs"]
        batch = data.select(batch_keys=select_keys).batch
        dataloader = batch.split(self.config.mini_batch_size)

        rm_scores_lst, q_lst = [], []

        for _, mini_batch in enumerate(dataloader):
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                micro_batches = mini_batch.split(self.config.micro_batch_size_per_gpu)
                self.gradient_accumulation = self.config.mini_batch_size // self.config.micro_batch_size_per_gpu

            for mb in micro_batches:
                mb = mb.cuda()
                prompt_length = mb["prompts"].shape[-1]

                rm_score, q = self._forward_micro_batch(mb, prompt_length)
                rm_scores_lst.append(rm_score)
                q_lst.append(q.detach())

        rm_scores = torch.cat(rm_scores_lst, dim=0)
        q = torch.concat(q_lst, dim=0)

        loss = compute_eto_loss_rm(
            data=data,
            token_level_scores=q,
            beta=beta,
        )
        append_to_dict(metrics, {"reward_model/eto_loss": float(loss.detach().item())})

        self.reward_optimizer.zero_grad()
        loss.backward()
        grad_norm = self._optimizer_step()
        append_to_dict(metrics, {"reward_model/grad_norm": float(grad_norm.detach().item())})

        rm_scores = self.istar_norm(rm_scores)
        metrics.update(
            {
                "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
                "reward_model/raw_reward": q.sum(dim=-1).mean().item(),
            }
        )

        return rm_scores, metrics
