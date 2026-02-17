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

import torch
import torch.distributed
from flash_attn.bert_padding import (index_first_axis, pad_input, rearrange,
                                     unpad_input)
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from istar.rm_utils import (compute_ce_dpo_loss_rm, compute_bt_loss_rm,
                          compute_detach_dpo_loss_rm, compute_eto_loss_rm)
from verl import DataProto
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import (get_reverse_idx,
                                         rearrange_micro_batches)
from verl.utils.ulysses import (gather_outpus_and_unpad,
                                ulysses_pad_and_slice_inputs)

__all__ = ["DataParallelPRIMERewardModel"]


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

    def istar_norm(self, token_level_scores):
        if self.config.istar_norm == "batch_norm":
            reverse_cumsum = torch.cumsum(token_level_scores.flip(dims=[1]), dim=-1).flip(dims=[1])
            token_level_scores = token_level_scores / (reverse_cumsum.abs().max() + 1e-6)
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
    #     # make sure we are in training mode
    #     self.reward_module.train()
    #     metrics = {}

    #     beta = self.config.model.get("beta_train", 0.05)

    #     select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "acc", "prompts", "old_log_probs"]

    #     for key in ["Q_bc", "acc_bc"]:
    #         if key in data.batch.keys():
    #             select_keys.append(key)

    #     batch = data.select(batch_keys=select_keys).batch
    #     # Split to make minibatch iterator for updating the actor
    #     # See PPO paper for details. https://arxiv.org/abs/1707.06347
    #     dataloader = batch.split(self.config.mini_batch_size)

    #     rm_scores_lst = []
    #     q_lst = []

    #     for batch_idx, data in enumerate(dataloader):
    #         # split batch into micro_batches
    #         mini_batch = data
    #         if self.config.use_dynamic_bsz:
    #             max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
    #             micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
    #         else:
    #             micro_batches = mini_batch.split(self.config.micro_batch_size_per_gpu)
    #             self.gradient_accumulation = self.config.mini_batch_size // self.config.micro_batch_size_per_gpu

    #         self.reward_optimizer.zero_grad()

    #         for data in micro_batches:
    #             data = data.cuda()
    #             attention_mask = data["attention_mask"]
    #             acc = data["acc"]

    #             prompt_ids = data["prompts"]
    #             prompt_length = prompt_ids.shape[-1]

    #             response_mask = attention_mask[:, prompt_length:]
                

    #             rm_score, q = self._forward_micro_batch(data, prompt_length)

    #             rm_scores_lst.append(rm_score)
    #             q_lst.append(q.detach())

    #             if self.config.model.loss_type == "ce":
    #                 dpo_loss = compute_ce_dpo_loss_rm(q, acc, response_mask=response_mask, beta=beta)
    #             elif self.config.model.loss_type == "dpo":
    #                 dpo_loss, num_pairs = compute_bt_loss_rm(token_level_scores=q,acc=data["acc"],uid=data.non_tensor_batch["uid"],response_mask=response_mask,beta=beta,)
    #                 # the implementation of dpo is actually detached, which means we have to know the average value of w/l reward before the update.
    #             elif self.config.model.loss_type == "bon_acc":
    #                 # change the original distribution of each sample to BoN distribution, then update reward model
    #                 dpo_loss = compute_detach_dpo_loss_rm(
    #                     q,
    #                     acc,
    #                     Q_bc=data["Q_bc"],
    #                     acc_bc=data["acc_bc"],
    #                     response_mask=response_mask,
    #                     beta=beta,
    #                     bon_mode="bon_acc",
    #                 )
    #             elif self.config.model.loss_type == "bon_rm":
    #                 dpo_loss = compute_detach_dpo_loss_rm(
    #                     q,
    #                     acc,
    #                     Q_bc=data["Q_bc"],
    #                     acc_bc=data["acc_bc"],
    #                     response_mask=response_mask,
    #                     beta=beta,
    #                     bon_mode="bon_rm",
    #                 )
    #             else:
    #                 raise NotImplementedError

    #             data = {"reward_model/dpo_loss": dpo_loss.detach().item()}

    #             if self.config.use_dynamic_bsz:
    #                 # relative to the dynamic bsz
    #                 # loss = dpo_loss * (len(data) / self.config.ppo_mini_batch_size)
    #                 mbsz = attention_mask.shape[0]
    #                 loss = dpo_loss * (mbsz / self.config.ppo_mini_batch_size)
    #             else:
    #                 loss = dpo_loss / self.gradient_accumulation

    #             loss.backward()

    #             append_to_dict(metrics, data)

    #         grad_norm = self._optimizer_step()
    #         data = {"reward_model/grad_norm": grad_norm.detach().item()}
    #         append_to_dict(metrics, data)
    #     self.reward_optimizer.zero_grad()

    #     rm_scores = torch.concat(rm_scores_lst, dim=0)
    #     q = torch.concat(q_lst, dim=0)

    #     rm_scores = self.istar_norm(rm_scores)

    #     metrics.update(
    #         {
    #             "reward_model/reward": rm_scores.sum(dim=-1).mean().item(),
    #             "reward_model/raw_reward": q.sum(dim=-1).mean().item(),
    #         }
    #     )

    #     return rm_scores, metrics

    def update_rm(self, data: DataProto):
        self.reward_module.train()
        metrics = {}

        beta = self.config.model.get("beta_train", 0.05)

        select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "acc", "prompts", "old_log_probs"]

        # --- NEW: ensure uid exists in tensor batch ---
        if "uid" not in data.batch.keys():
            # uid originally lives in non_tensor_batch
            import numpy as np
            uid0 = np.asarray(data.non_tensor_batch["uid"])
            # must be numeric; convert object->int64 safely
            uid0 = uid0.astype(np.int64)
            data.batch["uid"] = torch.from_numpy(uid0).long()
        # --- END NEW ---

        # make sure it gets selected
        select_keys.append("uid")

        for key in ["Q_bc", "acc_bc"]:
            if key in data.batch.keys():
                select_keys.append(key)

        batch = data.select(batch_keys=select_keys).batch
        dataloader = batch.split(self.config.mini_batch_size)

        rm_scores_lst = []
        q_lst = []

        for batch_idx, mini in enumerate(dataloader):
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

                elif self.config.model.loss_type == "dpo":
                    dpo_loss, num_pairs = compute_bt_loss_rm(
                        token_level_scores=q,
                        acc=acc,
                        uid=mb["uid"],                  # <-- now tensor uid exists
                        response_mask=response_mask,
                        beta=beta,
                    )

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
        metrics.update({
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
                    loss_val = compute_ce_dpo_loss_rm(q, acc, response_mask=response_mask, beta=beta)

                elif self.config.model.loss_type == "dpo":
                    loss_val, _num_pairs = compute_bt_loss_rm(
                        token_level_scores=q,
                        acc=acc,
                        uid=mb["uid"],
                        response_mask=response_mask,
                        beta=beta,
                    )

                elif self.config.model.loss_type == "bon_acc":
                    loss_val = compute_detach_dpo_loss_rm(
                        q,
                        acc,
                        Q_bc=mb["Q_bc"],
                        acc_bc=mb["acc_bc"],
                        response_mask=response_mask,
                        beta=beta,
                        bon_mode="bon_acc",
                    )

                elif self.config.model.loss_type == "bon_rm":
                    loss_val = compute_detach_dpo_loss_rm(
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

                append_to_dict(metrics, {"reward_model/dpo_loss": float(loss_val.detach().item())})

                if self.config.use_dynamic_bsz:
                    mbsz = attention_mask.shape[0]
                    loss = loss_val * (mbsz / self.config.ppo_mini_batch_size)
                else:
                    loss = loss_val / self.gradient_accumulation

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
