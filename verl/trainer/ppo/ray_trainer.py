# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os,math
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict, List, Union, Optional, Tuple, Any

import re
import json
from collections import defaultdict
from tqdm import tqdm

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance, karmarkar_karp

import re
from search_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]



# ---------------------------
# Helpers
# ---------------------------

def _ensure_uid_tensor(data) -> None:
    """
    Ensure data.batch["uid"] exists as torch.long of shape (B,).
    Uses data.non_tensor_batch["uid"] if needed.
    """
    if "uid" in data.batch:
        if data.batch["uid"].dtype != torch.long:
            data.batch["uid"] = data.batch["uid"].long()
        return

    if not hasattr(data, "non_tensor_batch") or "uid" not in data.non_tensor_batch:
        raise KeyError("uid not found in data.batch or data.non_tensor_batch")

    uid0 = np.asarray(data.non_tensor_batch["uid"])
    uid0 = uid0.astype(np.int64)
    data.batch["uid"] = torch.from_numpy(uid0).long()


def _to_cpu_detached(x: torch.Tensor, *, cast_mask_to_uint8: bool = True) -> torch.Tensor:
    """
    Store tensors on CPU to save GPU memory.
    Optionally cast attention_mask to uint8 to save memory.
    """
    x = x.detach().to("cpu")
    if cast_mask_to_uint8 and x.dtype == torch.long and x.dim() >= 1:
        # only cast masks if caller uses this appropriately
        pass
    return x

def _to_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu")

def _maybe_cast_attention_mask_uint8(row: Dict[str, torch.Tensor], enable: bool) -> None:
    if not enable:
        return
    if "attention_mask" in row and row["attention_mask"].dtype == torch.long:
        row["attention_mask"] = row["attention_mask"].to(torch.uint8)


def _restore_attention_mask_long(batch: Dict[str, torch.Tensor]) -> None:
    if "attention_mask" in batch and batch["attention_mask"].dtype == torch.uint8:
        batch["attention_mask"] = batch["attention_mask"].long()


# def _stack_rows(rows: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
#     if not rows:
#         raise ValueError("empty rows")
#     keys = rows[0].keys()
#     out = {}
#     for k in keys:
#         out[k] = torch.stack([r[k] for r in rows], dim=0)
#     return out

def _stack_rows(
    rows: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    if not rows:
        raise ValueError("empty rows")

    keys = rows[0].keys()
    out = {}

    for k in keys:
        vals = [r[k] for r in rows]
        shapes = [tuple(v.shape) for v in vals]

        if all(s == shapes[0] for s in shapes):
            out[k] = torch.stack(vals, dim=0)
            continue

        if all(v.dim() == 1 for v in vals):
            max_len = max(v.size(0) for v in vals)

            if k in ["input_ids", "responses"]:
                cur_pad_value = pad_token_id
            elif k in ["attention_mask", "response_mask", "loss_mask"]:
                cur_pad_value = 0
            elif k in ["position_ids"]:
                cur_pad_value = 0
            elif k in ["old_log_probs", "advantages", "returns", "values", "token_level_scores", "ref_log_prob"]:
                cur_pad_value = 0.0
            else:
                raise RuntimeError(f"Variable-length field {k} has no pad rule")

            padded = [
                F.pad(v, (0, max_len - v.size(0)), value=cur_pad_value)
                for v in vals
            ]
            out[k] = torch.stack(padded, dim=0)
            continue

        if all(v.dim() == 2 for v in vals):
            max0 = max(v.size(0) for v in vals)
            max1 = max(v.size(1) for v in vals)

            if k in ["input_ids", "responses"]:
                cur_pad_value = pad_token_id
            else:
                cur_pad_value = 0

            padded = []
            for v in vals:
                pad = (0, max1 - v.size(1), 0, max0 - v.size(0))
                padded.append(F.pad(v, pad, value=cur_pad_value))
            out[k] = torch.stack(padded, dim=0)
            continue

        raise RuntimeError(f"Cannot stack key={k}: inconsistent shapes={shapes}")

    return out

def _gather_rows(batch_tensors: Dict[str, torch.Tensor], idx: List[int]) -> Dict[str, torch.Tensor]:
    """
    Select rows from a dict of (B, ...) tensors.
    """
    out = {}
    idx_t = torch.tensor(idx, dtype=torch.long)
    for k, v in batch_tensors.items():
        if not torch.is_tensor(v):
            continue
        out[k] = v.index_select(0, idx_t)
    return out


# def _stack_items(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
#     """
#     Stack a list of per-row dicts into a batch dict.
#     Each item[k] should be a tensor of shape (...)
#     """
#     if not items:
#         raise ValueError("No items to stack")

#     keys = items[0].keys()
#     out: Dict[str, torch.Tensor] = {}
#     for k in keys:
#         vals = [it[k] for it in items]
#         out[k] = torch.stack(vals, dim=0)
#     return out

def _pad_and_stack_1d(vals, pad_value=0):
    # vals: list[Tensor] with shape [L_i]
    max_len = max(int(v.numel()) for v in vals)
    out = []
    for v in vals:
        v = v.detach().cpu()
        if v.numel() < max_len:
            pad = torch.full((max_len - v.numel(),), pad_value, dtype=v.dtype)
            v = torch.cat([v, pad], dim=0)
        out.append(v)
    return torch.stack(out, dim=0)

def _pad_and_stack_2d(vals, pad_value=0):
    # vals: list[Tensor] with shape [L_i, D] or [D, L_i] (we handle only [L_i, D])
    max_len = max(int(v.shape[0]) for v in vals)
    D = int(vals[0].shape[1])
    out = []
    for v in vals:
        v = v.detach().cpu()
        if v.shape[0] < max_len:
            pad = torch.full((max_len - v.shape[0], D), pad_value, dtype=v.dtype)
            v = torch.cat([v, pad], dim=0)
        out.append(v)
    return torch.stack(out, dim=0)

def _stack_items(flat_items, pad_token_id: int = 0):
    """
    flat_items: list[dict[str, Tensor]]
    Returns: dict[str, Tensor] stacked on dim=0.
    Pads variable-length tensors on the RIGHT.
    """
    if len(flat_items) == 0:
        return {}

    keys = flat_items[0].keys()
    out = {}

    # per-key pad values (adjust if needed)
    pad_value_by_key = {
        "input_ids": pad_token_id,
        "attention_mask": 0,
        "position_ids": 0,
        "responses": pad_token_id,
        "prompts": pad_token_id,
        "old_log_probs": 0.0,
    }

    for k in keys:
        vals = [it[k] for it in flat_items]

        # scalars / 1D fixed-size
        if vals[0].dim() == 0:
            out[k] = torch.stack([v.detach().cpu() for v in vals], dim=0)
            continue

        # 1D variable-length (most common: input_ids, attention_mask, position_ids, old_log_probs)
        if vals[0].dim() == 1:
            pad_val = pad_value_by_key.get(k, 0)
            out[k] = _pad_and_stack_1d(vals, pad_value=pad_val)
            continue

        # 2D variable-length (rare in your RM path, but keep it)
        if vals[0].dim() == 2 and vals[0].shape[1] == vals[0].shape[1]:
            pad_val = pad_value_by_key.get(k, 0)
            out[k] = _pad_and_stack_2d(vals, pad_value=pad_val)
            continue

        # fallback: require equal shapes
        out[k] = torch.stack([v.detach().cpu() for v in vals], dim=0)

    return out


def _maybe_cast_masks(batch: Dict[str, torch.Tensor], cast_attention_mask_to_uint8: bool) -> None:
    if not cast_attention_mask_to_uint8:
        return
    if "attention_mask" in batch and batch["attention_mask"].dtype == torch.long:
        # uint8 is plenty for 0/1 mask
        batch["attention_mask"] = batch["attention_mask"].to(torch.uint8)


def _restore_masks_for_model(batch: Dict[str, torch.Tensor]) -> None:
    """
    If your model expects attention_mask long/bool, adjust here.
    Many HF models accept int64 or bool; uint8 is usually OK, but
    safest is convert back to long.
    """
    if "attention_mask" in batch and batch["attention_mask"].dtype == torch.uint8:
        batch["attention_mask"] = batch["attention_mask"].long()


def _get_group_key(uid: int, iter_id: Optional[int]) -> Tuple[int, Optional[int]]:
    # Group by (uid, iter_id) if iter_id is provided; else group by uid only.
    return (uid, iter_id)


# ---------------------------
# Base: trajectory store item
# ---------------------------

@dataclass
class TrajItem:
    uid: int
    iter_id: Optional[int]  # for optional grouping by iteration
    tensors: Dict[str, torch.Tensor]  # per-row tensors on CPU

# ---------------------------
# Buffer 1: CE replay (all trajectories)
# ---------------------------

class CEReplayBuffer:
    """
    Stores individual trajectories (pos+neg) with labels (acc).
    For CE loss.
    """
    def __init__(
        self,
        capacity_traj: int,
        keys_to_store: List[str],
        cast_attention_mask_to_uint8: bool = True,
        seed: int = 0,
        pad_token_id: int = 0,
    ):
        self.pad_token_id = int(pad_token_id)
        self.capacity = int(capacity_traj)
        self.keys = list(keys_to_store)
        self.cast_mask = cast_attention_mask_to_uint8
        self.rng = random.Random(seed)
        self._items: List[TrajItem] = []

    def __len__(self) -> int:
        return len(self._items)

    def add(self, data, *, iter_id: Optional[int] = None) -> None:
        _ensure_uid_tensor(data)
        B = data.batch["uid"].shape[0]

        # select tensors to store
        store_batch: Dict[str, torch.Tensor] = {}
        for k in self.keys + ["uid"]:
            if k in data.batch:
                store_batch[k] = data.batch[k]
        _maybe_cast_masks(store_batch, self.cast_mask)

        uid_np = store_batch["uid"].to("cpu").numpy().astype(np.int64)

        for i in range(B):
            row = {}
            for k, v in store_batch.items():
                row[k] = _to_cpu_detached(v[i])
            uid_i = int(uid_np[i])
            self._items.append(TrajItem(uid=uid_i, iter_id=iter_id, tensors=row))

        # evict oldest
        overflow = len(self._items) - self.capacity
        if overflow > 0:
            self._items = self._items[overflow:]

    def sample(self, num_traj: int, dp_world_size: int = 1):
        if len(self._items) == 0:
            return None
        n = min(int(num_traj), len(self._items))
        if dp_world_size > 1:
            n = (n // dp_world_size) * dp_world_size
        if n == 0:
            return None
        idx = self.rng.sample(range(len(self._items)), k=n)
        items = [self._items[i].tensors for i in idx]
        batch = _stack_items(items, pad_token_id=self.pad_token_id)
        _restore_masks_for_model(batch)
        return batch


# ---------------------------
# Buffer 2: IRL replay
# ---------------------------

@dataclass
class GroupItem:
    uid: int
    iter_id: int
    # list of per-trajectory row dicts (CPU tensors)
    rows: List[Dict[str, torch.Tensor]]  # length = cands_per_uid

class GroupReplayBuffer:
    def __init__(
        self,
        capacity_groups: int,
        cands_per_uid: int,
        keys_to_store: List[str],
        cast_attention_mask_to_uint8: bool = True,
        seed: int = 0,
        acc_key: str = "acc",
        pos_threshold: float = 0.5,
        require_both_pos_neg: bool = True,
        pad_token_id: int = 0,
    ):
        self.pad_token_id = int(pad_token_id)
        self.capacity = int(capacity_groups)
        self.cands_per_uid = int(cands_per_uid)
        self.keys = list(keys_to_store)
        self.cast_mask = bool(cast_attention_mask_to_uint8)
        self.acc_key = acc_key
        self.pos_threshold = float(pos_threshold)
        self.require_both_pos_neg = bool(require_both_pos_neg)
        self.rng = random.Random(seed)

        self._groups: List[GroupItem] = []
        self._seen = set()  # (uid, iter_id)

    def add_from_batch(self, data, *, iter_id: int) -> int:
        _ensure_uid_tensor(data)
        if self.acc_key not in data.batch:
            raise KeyError(f"{self.acc_key} not found in data.batch")

        uid = data.batch["uid"].detach().to("cpu").long()
        acc = data.batch[self.acc_key].detach().to("cpu").float()
        B = uid.shape[0]

        store_batch: Dict[str, torch.Tensor] = {}
        for k in self.keys + ["uid"]:
            if k in data.batch:
                store_batch[k] = data.batch[k]
        if self.acc_key in data.batch and self.acc_key not in store_batch:
            store_batch[self.acc_key] = data.batch[self.acc_key]

        uid_np = uid.numpy().astype(np.int64)
        by_uid: Dict[int, List[int]] = {}
        for i in range(B):
            u = int(uid_np[i])
            by_uid.setdefault(u, []).append(i)

        added = 0
        for u, idxs in by_uid.items():
            # require exact group size
            if len(idxs) != self.cands_per_uid:
                continue

            # NEW: only keep groups with both pos and neg
            if self.require_both_pos_neg:
                acc_g = acc[idxs]
                has_pos = bool((acc_g > self.pos_threshold).any().item())
                has_neg = bool((acc_g <= self.pos_threshold).any().item())
                if not (has_pos and has_neg):
                    continue
            # END NEW

            key = (u, int(iter_id))
            if key in self._seen:
                continue

            rows = []
            for i in idxs:
                row = {k: _to_cpu(v[i]) for k, v in store_batch.items()}
                _maybe_cast_attention_mask_uint8(row, self.cast_mask)
                row["uid"] = torch.tensor(u, dtype=torch.long)
                rows.append(row)

            self._groups.append(GroupItem(uid=u, iter_id=int(iter_id), rows=rows))
            self._seen.add(key)
            added += 1

        # evict oldest
        overflow = len(self._groups) - self.capacity
        if overflow > 0:
            evicted = self._groups[:overflow]
            self._groups = self._groups[overflow:]
            for g in evicted:
                self._seen.discard((g.uid, g.iter_id))

        return added

    def sample_groups_as_flat_batch(self, num_groups: int) -> Optional[Dict[str, torch.Tensor]]:
        """
        Sample groups and return a flat trajectory batch (num_groups*cands_per_uid, ...).
        This is what your existing update_rm() expects.
        """
        if len(self._groups) == 0:
            return None
        n = min(int(num_groups), len(self._groups))
        idx = self.rng.sample(range(len(self._groups)), k=n)
        chosen = [self._groups[i] for i in idx]

        flat_rows: List[Dict[str, torch.Tensor]] = []
        for g in chosen:
            flat_rows.extend(g.rows)

        batch = _stack_rows(flat_rows, pad_token_id=self.pad_token_id)
        _restore_attention_mask_long(batch)

        # sanity: ensure the flat batch size is divisible by cands_per_uid
        assert batch["uid"].shape[0] % self.cands_per_uid == 0
        return batch


# ---------------------------
# Buffer 3: Preference-pair replay (BT/DPO)
# ---------------------------

@dataclass
class PairItem:
    uid: int
    iter_id: Optional[int]
    pos: Dict[str, torch.Tensor]  # per-row tensors on CPU
    neg: Dict[str, torch.Tensor]


class PairReplayBuffer:
    """
    Stores explicit (pos, neg) pairs for BT/DPO loss.
    This avoids the "no pairs" problem entirely.
    """
    def __init__(
        self,
        capacity_pairs: int,
        keys_to_store: List[str],
        cast_attention_mask_to_uint8: bool = True,
        seed: int = 0,
        acc_key: str = "acc",
        pos_threshold: float = 0.5,
        group_by_iter: bool = True,
        max_pairs_per_group: int = 4,   # <-- recommended cap
        pad_token_id: int = 0,
    ):
        self.pad_token_id = int(pad_token_id)
        self.capacity = int(capacity_pairs)
        self.keys = list(keys_to_store)
        self.cast_mask = cast_attention_mask_to_uint8
        self.rng = random.Random(seed)
        self.acc_key = acc_key
        self.pos_threshold = float(pos_threshold)
        self.group_by_iter = bool(group_by_iter)
        self.max_pairs_per_group = int(max_pairs_per_group)
        self._pairs: List[PairItem] = []

    def __len__(self) -> int:
        return len(self._pairs)

    def add_from_batch(self, data, *, iter_id: Optional[int] = None) -> int:
        """
        Create (pos, neg) pairs within each group and store them.
        Returns number of pairs added.
        """
        _ensure_uid_tensor(data)
        if self.acc_key not in data.batch:
            raise KeyError(f"{self.acc_key} not found in data.batch")

        uid = data.batch["uid"].detach().to("cpu").long()
        acc = data.batch[self.acc_key].detach().to("cpu").float()

        store_batch: Dict[str, torch.Tensor] = {}
        for k in self.keys + ["uid", self.acc_key]:
            if k in data.batch:
                store_batch[k] = data.batch[k]
        _maybe_cast_masks(store_batch, self.cast_mask)

        # group indices by (uid, iter_id?) to avoid mixing across iterations (recommended)
        groups: Dict[Tuple[int, Optional[int]], List[int]] = {}
        B = uid.shape[0]
        for i in range(B):
            u = int(uid[i].item())
            g_iter = iter_id if self.group_by_iter else None
            gk = _get_group_key(u, g_iter)
            groups.setdefault(gk, []).append(i)

        added = 0
        for (u, it), idxs in groups.items():
            # split into pos/neg indices
            pos = [i for i in idxs if float(acc[i].item()) > self.pos_threshold]
            neg = [i for i in idxs if float(acc[i].item()) <= self.pos_threshold]
            if not pos or not neg:
                continue

            # cap pairs per group to avoid skew
            k = min(self.max_pairs_per_group, len(pos), len(neg))
            # random match (without replacement)
            self.rng.shuffle(pos)
            self.rng.shuffle(neg)
            pos = pos[:k]
            neg = neg[:k]

            for pi, ni in zip(pos, neg):
                pos_row = {kk: _to_cpu_detached(store_batch[kk][pi]) for kk in store_batch.keys() if kk != "uid"}
                neg_row = {kk: _to_cpu_detached(store_batch[kk][ni]) for kk in store_batch.keys() if kk != "uid"}

                # uid stored separately
                self._pairs.append(PairItem(uid=u, iter_id=it, pos=pos_row, neg=neg_row))
                added += 1

        # evict oldest
        overflow = len(self._pairs) - self.capacity
        if overflow > 0:
            self._pairs = self._pairs[overflow:]

        return added

    def sample_pairs(self, num_pairs: int) -> Optional[List[PairItem]]:
        if len(self._pairs) == 0:
            return None
        n = min(int(num_pairs), len(self._pairs))
        idx = self.rng.sample(range(len(self._pairs)), k=n)
        return [self._pairs[i] for i in idx]

    def sample_as_flat_trajectories(self, num_pairs: int, dp_world_size: int = 1):
        if len(self._pairs) == 0:
            return None

        n = min(int(num_pairs), len(self._pairs))

        if dp_world_size > 1:
            # need (2*n) % dp_world_size == 0
            g = math.gcd(dp_world_size, 2)
            step = dp_world_size // g  # for dp=8 -> step=4
            n = (n // step) * step

        if n == 0:
            return None

        pairs = self.sample_pairs(n)
        if pairs is None:
            return None

        flat = []
        for p in pairs:
            pos_row = dict(p.pos)
            neg_row = dict(p.neg)
            pos_row["acc"] = torch.tensor(1.0, dtype=torch.float32)
            neg_row["acc"] = torch.tensor(0.0, dtype=torch.float32)
            pos_row["uid"] = torch.tensor(p.uid, dtype=torch.long)
            neg_row["uid"] = torch.tensor(p.uid, dtype=torch.long)

            # (recommended) contiguous [pos,neg] per pair
            flat.append(pos_row)
            flat.append(neg_row)

        batch = _stack_items(flat, pad_token_id=self.pad_token_id)
        _restore_masks_for_model(batch)
        return batch


# ---------------------------
# Glue: build DataProto for RM update
# ---------------------------

def build_dataproto_for_rm(DataProtoCls, tensors: Dict[str, torch.Tensor], *, meta_n: Optional[int] = None):
    """
    DataProtoCls: your DataProto class (pass DataProto).
    tensors: dict of (B, ...) tensors.
    meta_n: set meta_info["n"] for metrics that assume n samples per prompt.
    """
    meta = {}
    if meta_n is not None:
        meta["n"] = int(meta_n)
    return DataProtoCls.from_dict(tensors=tensors, meta_info=meta)


import torch
from verl.utils.torch_functional import masked_mean

def _pad_2d_right(x: torch.Tensor, target_len: int, pad_value):
    # x: (B, L)
    if x.size(1) == target_len:
        return x
    assert x.dim() == 2, f"expected 2D tensor, got {x.shape}"
    pad_len = target_len - x.size(1)
    assert pad_len > 0
    pad = torch.full((x.size(0), pad_len), pad_value, device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)

def pad_dataproto_to_match(outputs, pad_token_id: int):
    """
    outputs: List[DataProto], each has a TensorDict batch.
    Pads all 2D tensors (B, L) to the max L across outputs for that key.
    """
    # collect target lengths per key
    keys = list(outputs[0].batch.keys())
    target_len = {}
    for k in keys:
        # only pad 2D tensors
        if outputs[0].batch[k] is None:
            continue
        if not isinstance(outputs[0].batch[k], torch.Tensor):
            continue
        if outputs[0].batch[k].dim() != 2:
            continue
        target_len[k] = max(o.batch[k].size(1) for o in outputs)

    # choose pad values per key
    def pad_value_for(k, x):
        # heuristic: token ids / position ids use int pads, masks use 0, logprobs use 0.0
        if k in ("input_ids", "prompts", "responses"):
            return pad_token_id
        if k in ("attention_mask", "loss_mask", "info_mask"):
            return 0
        if k in ("position_ids", "turn_indices"):
            return 0
        if "log_prob" in k or "logprob" in k or k in ("old_log_probs", "ref_log_prob"):
            return 0.0
        # safe default: 0 of correct dtype kind
        return 0.0 if x.is_floating_point() else 0

    # pad each output
    for o in outputs:
        for k, L in target_len.items():
            x = o.batch[k]
            pv = pad_value_for(k, x)
            o.batch[k] = _pad_2d_right(x, L, pv)

    return outputs



def build_qbc_accbc(batch, num_rollout: int):
    """
    Aligns with the first-set behavior:
      - Q_bc, acc_bc have shape (N, R) where R=num_rollout
      - each row contains the full group vector INCLUDING self
      - all rows within a uid-group are identical (broadcast)
    """
    q_tok = batch.batch["q"]            # (N, Tq) token-level
    acc = batch.batch["acc"]            # (N,) binary {0,1} (float ok)
    uid = batch.non_tensor_batch["uid"] # length N object array

    # --- robust response mask aligned with q_tok length ---
    Tq = q_tok.size(1)
    attn = batch.batch["attention_mask"]
    response_mask = attn[:, -Tq:].float()          # (N, Tq)

    # scalar Q per candidate (NO beta here)
    q_seq = (q_tok * response_mask).sum(-1)        # (N,)

    # group indices by uid
    groups = defaultdict(list)
    # uid may be np.ndarray(dtype=object). Iterating yields python objects; keep as-is.
    for i, u in enumerate(uid):
        groups[u].append(i)

    N = q_seq.size(0)
    R = num_rollout

    Q_bc = torch.empty((N, R), device=q_seq.device, dtype=q_seq.dtype)
    acc_bc = torch.empty((N, R), device=acc.device, dtype=acc.dtype)

    # Build per-group vectors then broadcast to each member
    for u, idxs in groups.items():
        if len(idxs) != R:
            raise RuntimeError(
                f"uid={u} has {len(idxs)} candidates, expected {R}. "
                f"Check grouping: rollout.n / reward_model.num_rollout / repeats."
            )

        idx_t = torch.tensor(idxs, device=q_seq.device, dtype=torch.long)

        group_q = q_seq.index_select(0, idx_t)          # (R,)
        group_acc = acc.index_select(0, idx_t)          # (R,)

        # broadcast to all members (each row identical), matching first set
        for i in idxs:
            Q_bc[i] = group_q
            acc_bc[i] = group_acc

    batch.batch["Q_bc"] = Q_bc
    batch.batch["acc_bc"] = acc_bc
    return batch


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['info_mask'] if 'info_mask' in data.batch else data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def _debug_bt_uid_acc(batch, *, n_agent=None, n_roll=None, tag="pre_update_rm"):
    import numpy as np
    import torch

    print("\n" + "=" * 90)
    print(f"[BT-DEBUG] {tag}")
    print("-" * 90)

    # ---------- 0) Basic presence ----------
    assert "acc" in batch.batch, "[BT-DEBUG] missing batch.batch['acc']"
    assert "uid" in batch.non_tensor_batch, "[BT-DEBUG] missing batch.non_tensor_batch['uid']"

    acc_t = batch.batch["acc"].detach()
    if acc_t.is_cuda:
        acc_t_cpu = acc_t.float().cpu()
    else:
        acc_t_cpu = acc_t.float()

    uid_nt = np.asarray(batch.non_tensor_batch["uid"])
    try:
        uid_nt_i64 = uid_nt.astype(np.int64)
    except Exception as e:
        print("[BT-DEBUG] uid non_tensor cannot cast to int64:", type(uid_nt), uid_nt.dtype, "err:", repr(e))
        uid_nt_i64 = None

    # ---------- 1) Global uid stats (invariant A) ----------
    if uid_nt_i64 is not None:
        vals, cnts = np.unique(uid_nt_i64, return_counts=True)
        print(f"[BT-DEBUG] N={len(uid_nt_i64)} unique_uids={len(vals)} uid_count(min/mean/max)={cnts.min()}/{cnts.mean():.2f}/{cnts.max()}")
        if n_agent is not None and n_roll is not None:
            cands_per_prompt = int(n_agent) * int(n_roll)
            bad = int(np.sum(cnts != cands_per_prompt))
            print(f"[BT-DEBUG] expected cands_per_prompt={cands_per_prompt}  bad_uid_groups(count!=expected)={bad}")
        else:
            print("[BT-DEBUG] (n_agent/n_roll not provided) cannot check expected candidates per prompt.")

    # ---------- 2) uid tensor alignment check (invariant B) ----------
    if "uid" in batch.batch:
        uid_t = batch.batch["uid"].detach()
        uid_t_cpu = uid_t.long().cpu().numpy()
        if uid_nt_i64 is not None:
            print(f"[BT-DEBUG] uid tensor exists. matches non_tensor? {bool(np.array_equal(uid_t_cpu, uid_nt_i64))}")
        else:
            print("[BT-DEBUG] uid tensor exists, but non_tensor uid not int64-castable; skipping match check.")
    else:
        print("[BT-DEBUG] uid tensor NOT in batch.batch (it will be created inside update_rm in your code).")

    # ---------- 3) traj_uid uniqueness + pattern sanity (invariant C) ----------
    if "traj_uid" in batch.non_tensor_batch:
        traj = np.asarray(batch.non_tensor_batch["traj_uid"], dtype=object)
        uniq_traj = len(set(traj.tolist())) == len(traj)
        print(f"[BT-DEBUG] traj_uid present. unique? {uniq_traj}")
        if uid_nt_i64 is not None and len(traj) > 0:
            # optional pattern spot-check: "<uid>_c<k>"
            s = str(traj[0])
            ok_pat = ("_c" in s)
            print(f"[BT-DEBUG] traj_uid pattern spot-check (first='{s[:50]}...'): has '_c'? {ok_pat}")
    else:
        print("[BT-DEBUG] traj_uid not present (ok).")

    # ---------- 4) acc stats + per-uid label mixture (core, not using pair count) ----------
    acc_np = acc_t_cpu.numpy()
    print(f"[BT-DEBUG] acc stats: mean={float(acc_np.mean()):.4f} min={float(acc_np.min()):.4f} max={float(acc_np.max()):.4f} unique~={np.unique(acc_np)[:10].tolist()}{'...' if np.unique(acc_np).size>10 else ''}")

    # Compute per-uid mixture: both / only_pos / only_neg
    if uid_nt_i64 is not None:
        uid_t_for_group = torch.from_numpy(uid_nt_i64)
    else:
        # fallback: if no int64 uid, try to use uid tensor if present
        if "uid" in batch.batch:
            uid_t_for_group = batch.batch["uid"].detach().cpu().long()
        else:
            uid_t_for_group = None

    if uid_t_for_group is not None:
        uid_u = torch.unique(uid_t_for_group)
        both = only_pos = only_neg = 0
        for u in uid_u:
            idx = (uid_t_for_group == u).nonzero(as_tuple=False).squeeze(-1)
            a = acc_t_cpu[idx]
            has_pos = bool((a > 0.5).any().item())
            has_neg = bool((a <= 0.5).any().item())
            if has_pos and has_neg:
                both += 1
            elif has_pos:
                only_pos += 1
            else:
                only_neg += 1
        print(f"[BT-DEBUG] per-uid acc mixture: both={both} only_pos={only_pos} only_neg={only_neg} (total_uids={int(uid_u.numel())})")
    else:
        print("[BT-DEBUG] cannot compute per-uid mixture (no usable uid).")

    # ---------- 5) Microbatch-level “group split” check (what BT actually sees inside update_rm) ----------
    # We emulate your update_rm split behavior on CPU (no forward, just uid multiplicity per chunk).
    # IMPORTANT: this tests whether .split() will slice uid-groups apart.
    try:
        td = batch.select(batch_keys=["uid", "acc"]).batch if hasattr(batch, "select") else None
    except Exception:
        td = None

    if td is not None and "uid" in td:
        # mimic: dataloader = batch.split(mini_batch_size) then mini.split(micro_batch_size)
        mini_bsz = getattr(getattr(batch, "meta_info", {}), "get", lambda k, d=None: d)("mini_batch_size", None)
        print("[BT-DEBUG] Note: cannot read mini_batch_size from batch.meta_info; microbatch split test will use your config values if you pass them manually.")
    else:
        print("[BT-DEBUG] skip microbatch split test (cannot access tensordict 'uid'/'acc' here).")

    print("=" * 90)

def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'masked_gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = data.batch['loss_mask']
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_masked_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                             values=values,
                                                                             loss_mask=response_mask,
                                                                             gamma=gamma,
                                                                             lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'turn_level_gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        loss_mask = data.batch['loss_mask']
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_turn_level_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                                 values=values,
                                                                                 eos_mask=response_mask,
                                                                                 loss_mask=loss_mask,
                                                                                 gamma=gamma,
                                                                                 lam=lam,
                                                                                 turn_level_gamma=gamma,
                                                                                 turn_level_lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'turn_level_gae_v2':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        loss_mask = data.batch['loss_mask']
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_turn_level_gae_advantage_return_v2(token_level_rewards=token_level_rewards,
                                                                                 values=values,
                                                                                 eos_mask=response_mask,
                                                                                 loss_mask=loss_mask,
                                                                                 gamma=gamma,
                                                                                 lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'weighted_gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        loss_mask = data.batch['loss_mask']
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_weighted_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                               values=values,
                                                                               eos_mask=response_mask,
                                                                               loss_mask=loss_mask,
                                                                               gamma=gamma,
                                                                               lam=lam,                                                                                 
                                                                               turn_level_gamma=gamma,
                                                                               turn_level_lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'rloo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        response_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }

    # metrics for actions
    if 'turns_stats' in batch.meta_info:
        metrics['env/number_of_actions/mean'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).mean())
        metrics['env/number_of_actions/max'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).max())
        metrics['env/number_of_actions/min'] = float(np.array(batch.meta_info['turns_stats'], dtype=np.int16).min())
    if 'active_mask' in batch.meta_info:
        metrics['env/finish_ratio'] = 1 - float(np.array(batch.meta_info['active_mask'], dtype=np.int16).mean())
    if 'valid_action_stats' in batch.meta_info:
        metrics['env/number_of_valid_action'] = float(np.array(batch.meta_info['valid_action_stats'], dtype=np.int16).mean())
        metrics['env/ratio_of_valid_action'] = float((np.array(batch.meta_info['valid_action_stats'], dtype=np.int16) / np.array(batch.meta_info['turns_stats'], dtype=np.int16)).mean())
    if 'valid_search_stats' in batch.meta_info:
        metrics['env/number_of_valid_search'] = float(np.array(batch.meta_info['valid_search_stats'], dtype=np.int16).mean())


    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor', 'rollout']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()
        self._init_logger()

        ######################### replay buffer start ########################################
        cands_per_uid = int(self.config.actor_rollout_ref.rollout.n_agent) * int(self.config.actor_rollout_ref.rollout.n)

        rm_store_keys = ["input_ids", "attention_mask", "position_ids", "prompts", "responses", "acc", "old_log_probs"]

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        self.ce_buf = CEReplayBuffer(
            capacity_traj=200_000,
            keys_to_store=rm_store_keys,
            cast_attention_mask_to_uint8=True,
            seed=0,
            pad_token_id=pad_token_id
        )

        self.group_buf = GroupReplayBuffer(
            capacity_groups=50_000,              # groups, not trajectories
            cands_per_uid=cands_per_uid,         # 8 in your setting
            keys_to_store=rm_store_keys,
            cast_attention_mask_to_uint8=True,
            seed=1,
            acc_key="acc",
            pad_token_id=pad_token_id
        )

        self.pair_buf = PairReplayBuffer(
            capacity_pairs=200_000,
            keys_to_store=rm_store_keys,     # must include acc because we overwrite, but you can omit acc in store_keys too
            cast_attention_mask_to_uint8=True,
            seed=2,
            acc_key="acc",
            pos_threshold=0.5,
            group_by_iter=True,             # recommended: avoid cross-iteration mixing
            max_pairs_per_group=4,          # cap recommendation
            pad_token_id=pad_token_id
        )
    
        ######################### replay buffer end ########################################

    def _init_logger(self):
        from verl.utils.tracking import Tracking
        self.logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error')
        if self.config.data.train_data_num is not None:
            if self.config.data.train_data_num > len(self.train_dataset.dataframe):
                print(f"[WARNING] training dataset size is smaller than desired size. Using the dataset as the original size {len(self.train_dataset.dataframe)}")
            else:
                self.train_dataset.dataframe = self.train_dataset.dataframe.sample(self.config.data.train_data_num, random_state=42)
        print(f"filtered training dataset size: {len(self.train_dataset.dataframe)}")

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           shuffle=self.config.data.shuffle_train_dataloader,
                                           drop_last=True,
                                           collate_fn=collate_fn)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        if self.config.data.val_data_num is not None:
            if self.config.data.val_data_num > len(self.val_dataset.dataframe):
                print(f"[WARNING] validation dataset size is smaller than desired size. Using the dataset as the original size {len(self.val_dataset.dataframe)}")
            else:
                self.val_dataset.dataframe = self.val_dataset.dataframe.sample(self.config.data.val_data_num, random_state=42)
        print(f"filtered validation dataset size: {len(self.val_dataset.dataframe)}")

        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=self.config.data.val_batch_size,
                                         shuffle=False,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')
        
        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
        # if self.config.trainer.total_training_steps < total_training_steps:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        """
        The training loop of PPO with global metric computation.
        Accumulates metrics across all batches before computing final statistics.
        """
        import torch
        reward_tensor_lst = []
        answer_reward_tensor_lst = []
        answer_sub_em_reward_tensor_lst = []
        f1_score_reward_tensor_lst = []
        format_reward_tensor_lst = []
        retrieval_reward_tensor_lst = []
        mixed_outcome_reward_tensor_lst = []
        final_em_format_reward_tensor_lst = []
        avg_step_retrieval_format_reward_tensor_lst = []
        data_source_lst = []

        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
        )

        # Agent config preparation
        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation = True,
        )

        if not self.config.do_search:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print('validation generation end')

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                # for certain reward function (e.g. sandbox), the generation can overlap with reward
                reward_tensor = self.val_reward_fn(test_batch)

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
        else:
            # Prepare save directory once outside the loop
            save_dir = None
            if self.config.trainer.get('is_save_val_traj', False):
                # from datetime import datetime
                # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 20250730_153045
                val_traj_dir = self.config.trainer.get('val_traj_dir', './outputs/log_val_traj')
                save_dir = os.path.join(val_traj_dir, f"{self.config.trainer.experiment_name}")
                os.makedirs(save_dir, exist_ok=True)
            
            for i, batch_dict in enumerate(self.val_dataloader):
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)
                
                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }
                with _timer('step', timing_raw):
                    first_input_ids = test_gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone()
                    with _timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            initial_input_ids=first_input_ids,
                        )
                    
                    test_batch = test_batch.union(final_gen_batch_output)
                    
                    for key in test_batch.batch.keys():
                        test_batch.batch[key] = test_batch.batch[key].long()
                    
                    test_batch, _ = self._create_loss_mask(test_batch, {})
                    test_batch = self._split_turn_idx(test_batch)
                    
                    test_batch = self._split_trajectories(test_batch, save_dir, val_batch_idx=i)
                    
                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    reward_dict = self.val_reward_fn(test_batch)
                    answer_reward_tensor = reward_dict['answer_correctness']
                    answer_sub_em_reward_tensor = reward_dict['answer_sub_em']
                    f1_score_reward_tensor = reward_dict['f1_score']
                    format_reward_tensor = reward_dict['format_correctness']
                    retrieval_reward_tensor = reward_dict['retrieval_correctness']
                    mixed_outcome_reward_tensor = reward_dict['mixed_outcome_reward']
                    final_em_format_reward_tensor = reward_dict['final_em_format']
                    avg_step_retrieval_format_reward_tensor = reward_dict['avg_step_retrieval_format']
                    
                    answer_reward_tensor_lst.append(answer_reward_tensor)
                    answer_sub_em_reward_tensor_lst.append(answer_sub_em_reward_tensor)
                    f1_score_reward_tensor_lst.append(f1_score_reward_tensor)
                    format_reward_tensor_lst.append(format_reward_tensor)
                    retrieval_reward_tensor_lst.append(retrieval_reward_tensor)
                    mixed_outcome_reward_tensor_lst.append(mixed_outcome_reward_tensor)
                    final_em_format_reward_tensor_lst.append(final_em_format_reward_tensor)
                    avg_step_retrieval_format_reward_tensor_lst.append(avg_step_retrieval_format_reward_tensor)
                    data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * answer_reward_tensor.shape[0]))

        # reward_tensor = torch.cat([rw.sum(-1) for rw in reward_tensor_lst], dim=0).cpu()  # (batch_size,)
        # reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        answer_reward_tensor = torch.cat([rw.sum(-1, keepdim=True) for rw in answer_reward_tensor_lst], dim=0).cpu()
        answer_sub_em_reward_tensor = torch.cat([rw.sum(-1, keepdim=True) for rw in answer_sub_em_reward_tensor_lst], dim=0).cpu()
        f1_score_reward_tensor = torch.cat([rw.sum(-1, keepdim=True) for rw in f1_score_reward_tensor_lst], dim=0).cpu()
        format_reward_tensor = torch.cat([rw.sum(-1, keepdim=True) for rw in format_reward_tensor_lst], dim=0).cpu()
        retrieval_reward_tensor = torch.cat([rw.sum(-1, keepdim=True) for rw in retrieval_reward_tensor_lst], dim=0).cpu()
        mixed_outcome_reward_tensor = torch.cat([rw.sum(-1, keepdim=True) for rw in mixed_outcome_reward_tensor_lst], dim=0).cpu()
        final_em_format_reward_tensor = torch.cat([rw.sum(-1, keepdim=True) for rw in final_em_format_reward_tensor_lst], dim=0).cpu()
        avg_step_retrieval_format_reward_tensor = torch.cat([rw.sum(-1, keepdim=True) for rw in avg_step_retrieval_format_reward_tensor_lst], dim=0).cpu()

        data_sources = np.concatenate(data_source_lst, axis=0)

        metric_dict = {}
        metric_dict.update(self._track_reward_metrics(answer_reward_tensor, data_sources, prefix="val/test_score"))
        metric_dict.update(self._track_reward_metrics(answer_sub_em_reward_tensor, data_sources, prefix="val/answer_sub_em_score"))
        metric_dict.update(self._track_reward_metrics(f1_score_reward_tensor, data_sources, prefix="val/f1_score"))
        metric_dict.update(self._track_reward_metrics(format_reward_tensor, data_sources, prefix="val/format_score"))
        metric_dict.update(self._track_reward_metrics(retrieval_reward_tensor, data_sources, prefix="val/retrieval_score"))
        metric_dict.update(self._track_reward_metrics(mixed_outcome_reward_tensor, data_sources, prefix="val/mixed_outcome_score"))
        metric_dict.update(self._track_reward_metrics(final_em_format_reward_tensor, data_sources, prefix="val/final_em_format_score"))
        metric_dict.update(self._track_reward_metrics(avg_step_retrieval_format_reward_tensor, data_sources, prefix="val/avg_step_retrieval_format_score"))

        return metric_dict


    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae' or self.config.algorithm.adv_estimator == 'masked_gae' or \
                self.config.algorithm.adv_estimator == 'turn_level_gae' or self.config.algorithm.adv_estimator == 'weighted_gae' or \
                self.config.algorithm.adv_estimator == 'turn_level_gae_v2':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
            
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'rloo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)


    def _balance_batch_group_aware(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """
        Reorder data on the controller such that each DP rank gets similar total tokens,
        BUT keep samples with the same uid together (e.g., 8-per-uid group stays intact).

        This guarantees that later sequential chunking (DataProto.chunk(world_size)) will
        not split a uid-group across ranks.
        """
        import torch
        from collections import OrderedDict
        import numpy as np

        attention_mask = batch.batch['attention_mask']
        B = attention_mask.shape[0]
        seqlen = attention_mask.view(B, -1).sum(-1).tolist()

        world_size = self.actor_rollout_wg.world_size

        uid = batch.non_tensor_batch.get("uid", None)
        if uid is None:
            # fall back to your old behavior if no uid
            return self._balance_batch(batch, metrics, logging_prefix)

        uid_list = uid.tolist() if hasattr(uid, "tolist") else list(uid)

        # ---- build uid -> list of sample indices (preserve current within-uid order) ----
        groups = OrderedDict()
        for i, u in enumerate(uid_list):
            groups.setdefault(u, []).append(i)

        group_keys = list(groups.keys())
        group_indices = [groups[u] for u in group_keys]
        num_groups = len(group_indices)

        # ---- sanity: ensure group size is uniform (your requirement: 8) ----
        sizes = [len(idxs) for idxs in group_indices]
        if min(sizes) != max(sizes):
            # can't safely keep groups intact if inconsistent; fall back
            return self._balance_batch(batch, metrics, logging_prefix)

        group_size = sizes[0]
        # If you *require* exactly 8:
        if group_size != self.config.actor_rollout_ref.rollout.n_agent:
            return self._balance_batch(batch, metrics, logging_prefix)

        if num_groups < world_size:
            # not enough groups to distribute
            return self._balance_batch(batch, metrics, logging_prefix)

        # ---- compute group weights (sum of token lengths in the group) ----
        group_weights = [sum(seqlen[i] for i in idxs) for idxs in group_indices]

        # ---- partition groups across ranks ----
        # equal_size=True only if groups divide evenly across ranks
        equal_groups = (num_groups % world_size == 0)
        group_partitions = karmarkar_karp(
            seqlen_list=group_weights,
            k_partitions=world_size,
            equal_size=equal_groups,
        )
        # group_partitions: List[List[group_id]]

        # ---- build reordered sample index list: rank0 groups then rank1 ... ----
        new_order = []
        for part in group_partitions:
            for gid in sorted(part):   # optional: stable within-part
                new_order.extend(group_indices[gid])

        global_idx = torch.tensor(new_order, device=attention_mask.device, dtype=torch.long)
        batch.reorder(global_idx)

        # ---- metrics: how balanced are sums after group-aware partition ----
        balanced_group_sums = [sum(group_weights[gid] for gid in part) for part in group_partitions]
        metrics.update({
            f'{logging_prefix}/group_balanced_min': float(min(balanced_group_sums)),
            f'{logging_prefix}/group_balanced_max': float(max(balanced_group_sums)),
            f'{logging_prefix}/group_balanced_minmax_diff': float(max(balanced_group_sums) - min(balanced_group_sums)),
            f'{logging_prefix}/group_size': float(group_size),
        })

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """

        logger = self.logger
        self.global_steps = 0
        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1

        # Agent config preparation
        gen_config = GenerationConfig(
            max_turns=self.config.max_turns,
            max_start_length=self.config.data.max_start_length,
            max_prompt_length=self.config.data.max_prompt_length,
            max_response_length=self.config.data.max_response_length,
            max_obs_length=self.config.data.max_obs_length,
            num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
            no_think_rl=self.config.algorithm.no_think_rl,
            search_url = self.config.retriever.url,
            topk = self.config.retriever.topk,
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
        )

        save_dir = None
        if self.config.trainer.get('is_save_train_traj', False):
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # 20250730_153045
            train_traj_dir = self.config.trainer.get('train_traj_dir', './outputs/log_train_traj')
            save_dir = os.path.join(train_traj_dir, f"{self.config.trainer.experiment_name}_{timestamp}")
            os.makedirs(save_dir, exist_ok=True)


        # start training loop
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f'epoch {epoch}, step {self.global_steps}')
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n_agent, interleave=True)

                # pop those keys for generation
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                ####################
                # original code here

                with _timer('step', timing_raw):
                    if not self.config.do_search:

                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                dtype=object)
                        # repeat to align with repeated responses in rollout
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        
                        batch = batch.union(gen_batch_output)

                ####################
                # Below is aLL about agents - the "LLM + forloop"
                ####################
                # with _timer('step', timing_raw):
                    else:
                        first_input_ids = gen_batch.batch['input_ids'][:, -gen_config.max_start_length:].clone().long()

                        with _timer('gen', timing_raw):
                            generation_manager.timing_raw = timing_raw
                            final_gen_batch_output = generation_manager.run_llm_loop(
                                gen_batch=gen_batch,
                                initial_input_ids=first_input_ids,
                            )

                        # final_gen_batch_output.batch.apply(lambda x: x.long(), inplace=True)
                        for key in final_gen_batch_output.batch.keys():
                            final_gen_batch_output.batch[key] = final_gen_batch_output.batch[key].long()

                        with torch.no_grad():
                            output = self.actor_rollout_wg.compute_log_prob(final_gen_batch_output)
                            final_gen_batch_output = final_gen_batch_output.union(output)



                        n_agent = int(self.config.actor_rollout_ref.rollout.n_agent)
                        n_roll  = int(self.config.actor_rollout_ref.rollout.n)

                        # group id per prompt (already repeated by n_agent earlier)
                        uid0 = batch.non_tensor_batch["index"].copy()
                        batch.non_tensor_batch["uid"] = uid0  # keep group id

                        # if you really want rollout.n repetition here
                        if n_roll > 1:
                            batch = batch.repeat(repeat_times=n_roll, interleave=True)

                        uid = batch.non_tensor_batch["uid"]
                        total = len(uid)

                        # candidate index within each prompt-group
                        cands_per_prompt = n_agent * n_roll
                        assert total % cands_per_prompt == 0
                        candidate_idx = np.tile(np.arange(cands_per_prompt), total // cands_per_prompt).astype(int)

                        batch.non_tensor_batch["traj_uid"] = np.array(
                            [f"{u}_c{c}" for (u, c) in zip(uid.tolist(), candidate_idx.tolist())],
                            dtype=object
                        )

                        batch.non_tensor_batch["active_masks"] = np.array([True] * total, dtype=object)


                        batch = batch.union(final_gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    # self._balance_batch_group_aware(batch, metrics=metrics)
                    # self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # batch.batch.apply(lambda x, key: x.long() if key != "old_log_probs" else x, inplace=True, key=True)
                    for key in batch.batch.keys():
                        if key != 'old_log_probs':
                            batch.batch[key] = batch.batch[key].long()

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    if self.config.do_search and self.config.actor_rollout_ref.actor.state_masking:
                        batch, metrics = self._create_loss_mask(batch, metrics)
                        batch = self._split_turn_idx(batch)


                    batch = self._split_trajectories(batch, save_dir)

                    
                    with _timer('adv', timing_raw):
                        
                        # we combine with rule-based rm
                        reward_dict = self.reward_fn(batch)
                        if "acc" not in batch.batch:
                            acc_scalar = reward_dict["answer_correctness"].sum(dim=-1)  # (bs,)
                            batch.batch["acc"] = (acc_scalar > 0).float()


                        # RM update (NEW)
                        if self.use_rm:
                            rm_out = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(rm_out)
                            update_freq = self.config.reward_model.get("update_freq", 1)
                            # if self.config.reward_model.model.loss_type != "ce":
                            #     batch = build_qbc_accbc(batch, num_rollout=self.config.reward_model.num_rollout)

                            iter_id = int(self.global_steps)
                            self.ce_buf.add(batch, iter_id=iter_id)
                            self.group_buf.add_from_batch(batch, iter_id=iter_id)
                            self.pair_buf.add_from_batch(batch, iter_id=iter_id)
                            dp_world_size = self.config.trainer.n_gpus_per_node

                            if self.global_steps % update_freq == 0:
                                if self.config.reward_model.model.loss_type == "eto":
                                    rm_scores_upd = self.rm_wg.update_rm_eto(batch)
                                elif self.config.reward_model.model.loss_type == "ce":
                                    rm_scores_upd = self.rm_wg.update_rm(batch)
                                    # tensors = self.ce_buf.sample(num_traj=self.config.data.train_batch_size,dp_world_size=dp_world_size)
                                    # if tensors is not None:
                                    #     data = DataProto.from_dict(tensors=tensors, meta_info={"batch_layout": "ungrouped"})
                                    #     rm_scores_upd = self.rm_wg.update_rm(data)
                                elif self.config.reward_model.model.loss_type == "dpo":
                                    rm_scores_upd = self.rm_wg.update_rm(batch)
                                    # tensors = self.pair_buf.sample_as_flat_trajectories(num_pairs=int(self.config.data.train_batch_size/2),dp_world_size=dp_world_size)  # 512 trajectories
                                    # if tensors is not None:
                                    #     data = DataProto.from_dict(tensors=tensors, meta_info={"n_samples": 2, "batch_layout": "contiguous_pairs"})
                                    #     rm_scores_upd = self.rm_wg.update_rm(data)
                                elif self.config.reward_model.model.loss_type == "irl":
                                    tensors = self.group_buf.sample_groups_as_flat_batch(num_groups=int(self.config.data.train_batch_size/cands_per_prompt))
                                    if tensors is not None:
                                        data = DataProto.from_dict(tensors=tensors, meta_info={"n_samples": cands_per_prompt, "batch_layout": "contiguous_groups"})
                                        rm_scores_upd = self.rm_wg.update_rm(data)
                                else:
                                    rm_scores_upd = self.rm_wg.update_rm(batch)
                                    

                                    

                        # if self.use_rm:
                        #     # we first compute reward model score
                        #     reward_tensor = self.rm_wg.compute_rm_score(batch)
                        #     batch = batch.union(reward_tensor)
                            
                        
                        # Extract all reward tensors
                        answer_reward_tensor = reward_dict['answer_correctness']
                        answer_sub_em_reward_tensor = reward_dict['answer_sub_em']
                        f1_score_reward_tensor = reward_dict['f1_score']
                        format_reward_tensor = reward_dict['format_correctness']
                        retrieval_reward_tensor = reward_dict['retrieval_correctness']
                        mixed_outcome_reward_tensor = reward_dict['mixed_outcome_reward']
                        final_em_format_reward_tensor = reward_dict['final_em_format']
                        avg_step_retrieval_format_reward_tensor = reward_dict['avg_step_retrieval_format']
                        turn_level_reward_tensor = reward_dict['turn_level_reward']

                        # Get reward type and set token_level_scores accordingly
                        reward_type = self.config.algorithm.get('reward_type', 'outcome_reward')
                        print(f"[INFO] reward_type: {reward_type}")
                        
                        # Reward type mapping
                        reward_mapping = {
                            'outcome_reward': answer_reward_tensor,
                            'merged_reward': mixed_outcome_reward_tensor,
                            'turn_reward': turn_level_reward_tensor,
                        }
                        
                        # Handle special case for mixed_judge_reward
                        if 'judge' in reward_type:
                            judge_outcome_reward_tensor = reward_dict['judge_outcome_reward']
                            reward_mapping.update({
                                'judge_outcome_reward': judge_outcome_reward_tensor,
                            })
                            if reward_type == 'judge_turn_reward':
                                judge_turn_reward_tensor = reward_dict['judge_turn_reward']
                                reward_mapping.update({
                                    'judge_turn_reward': judge_turn_reward_tensor,
                                })
                        
                        # Set token_level_scores based on reward_type
                        if reward_type in reward_mapping and self.use_rm:
                            batch.batch["token_level_scores"] = batch.batch["rm_scores"] #+ format_reward_tensor
                            print(f"[INFO] Using reward shaping to shape {reward_type} for token_level_scores")
                        elif reward_type in reward_mapping:
                            selected_tensor = reward_mapping[reward_type]
                            print(f"[INFO] Using {reward_type} for token_level_scores")
                            batch.batch['token_level_scores'] = selected_tensor
                        
                        else:
                            raise ValueError(f"Unknown reward_type: {reward_type}. Valid options are: {list(reward_mapping.keys())}")
                            

                        # compute training reward metrics by data source
                        train_data_sources = batch.non_tensor_batch.get(
                            'data_source', ['unknown'] * answer_reward_tensor.shape[0]
                        )
                        
                        train_metric_dict = {}
                        train_metric_dict.update(self._track_reward_metrics(answer_reward_tensor, train_data_sources, prefix="train/reward"))
                        train_metric_dict.update(self._track_reward_metrics(answer_sub_em_reward_tensor, train_data_sources, prefix="train/answer_sub_em_reward"))
                        train_metric_dict.update(self._track_reward_metrics(f1_score_reward_tensor, train_data_sources, prefix="train/f1_score_reward"))
                        train_metric_dict.update(self._track_reward_metrics(format_reward_tensor, train_data_sources, prefix="train/format_reward"))
                        train_metric_dict.update(self._track_reward_metrics(retrieval_reward_tensor, train_data_sources, prefix="train/retrieval_reward"))
                        train_metric_dict.update(self._track_reward_metrics(mixed_outcome_reward_tensor, train_data_sources, prefix="train/mixed_outcome_reward"))
                        train_metric_dict.update(self._track_reward_metrics(final_em_format_reward_tensor, train_data_sources, prefix="train/final_em_format_reward"))
                        train_metric_dict.update(self._track_reward_metrics(avg_step_retrieval_format_reward_tensor, train_data_sources, prefix="train/avg_step_retrieval_format_reward"))
                        if 'judge' in reward_type:
                            train_metric_dict.update(self._track_reward_metrics(judge_outcome_reward_tensor, train_data_sources, prefix="train/judge_outcome_reward"))
                        if self.use_rm:
                            rm_metrics = rm_scores_upd.meta_info.get("metrics", {})
                            train_metric_dict.update(rm_metrics)

                        metrics.update(train_metric_dict)
                        logger.log(data=train_metric_dict, step=self.global_steps)

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']


                        ##############################debug_start###########################
                        # import collections
                        # uid = batch.non_tensor_batch["uid"]
                        # uid_list = uid.tolist() if hasattr(uid, "tolist") else list(uid)
                        # cnt = collections.Counter(uid_list)
                        # bad = [(u,c) for (u,c) in cnt.items() if c != 8]   # since n_agent=8, n_roll=1
                        # print("num_uids:", len(cnt), "bad_groups:", len(bad), "example:", bad[:10])
                        # input()
                        ##############################debug_end###########################

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    
                    progress_bar.close()
                    
                    return
    
    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens."""
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask

        metrics.update({
            'state_tokens/total': loss_mask.sum().item(),
            'state_tokens/coverage': (loss_mask.sum() / response_mask.sum()).item(),
        })
        
        return batch, metrics

    ########################################################################################################
    # def _split_turn_idx(self, batch: DataProto) -> DataProto:
    #     loss_mask = batch.batch['loss_mask']        
    #     # values = batch.batch['values']

    #     turn_indices = []

    #     for b in range(loss_mask.size(0)):
    #         mask = loss_mask[b]
    #         # valid_response_length = values[b].nonzero(as_tuple=True)[0].shape[0] - 1
    #         nonzero_indices = mask.nonzero(as_tuple=True)[0]
    #         if nonzero_indices.size(0) == 0:
    #             valid_response_length = 0
    #         else:
    #             valid_response_length = nonzero_indices[-1] + 1


    #         # Detect where a turn starts: when mask switches from 0 to 1
    #         turn_end_pos = ((mask[1:] == 1) & (mask[:-1] == 0)).nonzero(as_tuple=True)[0]
    #         turn_start_pos = turn_end_pos + 1

    #         # Check if the very first token is part of a turn
    #         if mask[0] == 1:
    #             turn_start_pos = torch.cat([torch.tensor([0], device=mask.device), turn_start_pos])

    #         # Append last token as final turn end if not already included

    #         turn_end_pos = torch.cat([turn_end_pos, torch.tensor([valid_response_length - 1], device=mask.device)])

    #         # Build list of (start, end) pairs
    #         indices = list(zip(turn_start_pos.tolist(), turn_end_pos.tolist()))
    #         turn_indices.append(indices)

    #     # Save to batch meta_info for later use (e.g., in GAE)
    #     batch.meta_info['turn_indices'] = turn_indices

    #     batch_size = len(turn_indices)
    #     max_indices = 20  # Should be enough for most cases (3 turns = max 6 indices, with buffer)
    #     turn_indices_tensor = torch.full((batch_size, max_indices), -1, dtype=torch.long, device=loss_mask.device)
        
    #     # Fill in the actual turn indices for each sample
        
    #     for b, indices in enumerate(turn_indices):
    #         flattened_indices = []
    #         for start, end in indices:
    #             flattened_indices.extend([start, end])
            
    #         # Fill the tensor with actual indices
    #         num_indices = min(len(flattened_indices), max_indices)
    #         turn_indices_tensor[b, :num_indices] = torch.tensor(flattened_indices[:num_indices], dtype=torch.long, device=loss_mask.device)
        
    #     batch.batch['turn_indices'] = turn_indices_tensor
        
    #     return batch

    def _split_turn_idx(self, batch: DataProto) -> DataProto:
        """
        Build per-sample (start, end) turn spans from loss_mask, where each turn is a
        contiguous run of loss_mask == 1 (within the response slice).

        Saves:
        - batch.meta_info['turn_indices']: List[List[Tuple[int,int]]]
        - batch.batch['turn_indices']: (B, max_indices) flattened [s0,e0,s1,e1,...] padded with -1
        """
        loss_mask = batch.batch['loss_mask']  # (B, T)
        device = loss_mask.device
        B, T = loss_mask.shape

        turn_indices: list[list[tuple[int, int]]] = []

        for b in range(B):
            m = loss_mask[b].to(torch.bool)  # (T,)

            # Positions where mask == 1
            idx = torch.nonzero(m, as_tuple=True)[0]
            if idx.numel() == 0:
                turn_indices.append([])
                continue

            # Breaks between non-consecutive indices -> segment boundaries
            # breaks contains positions in idx where a new segment starts at idx[break+1]
            breaks = torch.nonzero(idx[1:] != (idx[:-1] + 1), as_tuple=True)[0]

            # Segment starts/ends (inclusive)
            starts = torch.cat([idx[:1], idx[breaks + 1]])
            ends   = torch.cat([idx[breaks], idx[-1:]])

            # Convert to python list of tuples
            turn_indices.append(list(zip(starts.tolist(), ends.tolist())))

        # Save list form for decoding/logging
        batch.meta_info['turn_indices'] = turn_indices

        # Also store a fixed-size tensor form for convenience (optional)
        # Flatten each sample's (s,e) pairs into [s0,e0,s1,e1,...], pad with -1.
        # Keep your previous convention: max_indices=20.
        max_indices = 20
        turn_indices_tensor = torch.full((B, max_indices), -1, dtype=torch.long, device=device)

        for b, spans in enumerate(turn_indices):
            flat = []
            for s, e in spans:
                flat.extend([s, e])
            if len(flat) == 0:
                continue
            n = min(len(flat), max_indices)
            turn_indices_tensor[b, :n] = torch.tensor(flat[:n], dtype=torch.long, device=device)

        batch.batch['turn_indices'] = turn_indices_tensor
        return batch

    def _track_reward_metrics(self, reward_tensor: torch.Tensor, data_sources: List[str], prefix: str) -> Dict[str, float]:

        if reward_tensor.dim() == 0:
            reward_tensor = reward_tensor.unsqueeze(0)

        reward_tensor_sum = reward_tensor.sum(-1).cpu()
        data_source_reward_map = {}

        for i in range(reward_tensor_sum.shape[0]):
            source = data_sources[i]
            if source not in data_source_reward_map:
                data_source_reward_map[source] = []
            data_source_reward_map[source].append(reward_tensor_sum[i].item())

        metric_dict = {}
        for source, reward_list in data_source_reward_map.items():
            metric_dict[f"{prefix}/{source}"] = np.mean(reward_list)

        return metric_dict
    
    def _split_trajectories(self, batch, save_dir: Optional[str] = None, val_batch_idx: Optional[int] = None) -> DataProto:
        """
        Decode full trajectories and per-turn sequences from the batch, and store them
        into batch.meta_info for later use.

        Optionally, save them to disk if `save_dir` is provided.
        """
        full_texts = []
        prompt_texts = []
        turn_texts = []
        trajectories = []
        
        response_text_lengths = [] 
        turn_text_lengths = []
        num_turns = []

        turn_indices = batch.meta_info.get("turn_indices", [[] for _ in range(len(batch))])

        for i in range(len(batch)):
            data_item = batch[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            attention_mask = data_item.batch['attention_mask']
            valid_prompt_length = attention_mask[:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            prompt_text = self.tokenizer.decode(valid_prompt_ids)
            prompt_texts.append(prompt_text)
            
            response_ids = data_item.batch['responses']
            valid_response_length = attention_mask[prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            full_ids = torch.cat((valid_prompt_ids, valid_response_ids))
            full_text = self.tokenizer.decode(full_ids)
            full_texts.append(full_text)
            response_text_lengths.append(valid_response_ids.shape[0])

            # Turn-level decoding
            turns = []
            turn_lengths = []
            for start, end in turn_indices[i]:
                turn_ids = response_ids[start:end + 1]
                turn_text = self.tokenizer.decode(turn_ids)
                turns.append(turn_text)
                turn_lengths.append(turn_ids.shape[0])
            turn_texts.append(turns)
            turn_text_lengths.append(turn_lengths)
            num_turns.append(len(turns))

            # Optional for logging/saving
            ground_truth = data_item.non_tensor_batch.get('reward_model', {}).get('ground_truth', {}).get('target', '')
            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')
            if isinstance(ground_truth, np.ndarray):
                ground_truth = ground_truth.tolist()
            if isinstance(data_source, np.ndarray):
                data_source = data_source.tolist()

            trajectories.append({
                "data_source": data_source,
                "ground_truth": ground_truth,
                "full_text": full_text,
                "prompt": prompt_text,
                "turn_texts": turns,
            })

        # Inject into batch.meta_info
        batch.meta_info["decoded_full_texts"] = full_texts
        batch.meta_info["decoded_prompts"] = prompt_texts
        batch.meta_info["decoded_turn_texts"] = turn_texts
        batch.meta_info["response_text_lengths"] = response_text_lengths
        batch.meta_info["turn_text_token_lengths"] = turn_text_lengths
        batch.meta_info["num_turns"] = num_turns

        # Optional: save to JSON
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            if val_batch_idx is not None:
                save_path = os.path.join(save_dir, f'trajectories_val_batch_{val_batch_idx}.json')
            else:
                save_path = os.path.join(save_dir, f'trajectories_step_{self.global_steps}.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(trajectories, f, ensure_ascii=False, indent=2)

        return batch
