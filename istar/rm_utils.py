import random

import torch
from collections import defaultdict
import verl
from verl import DataProto
import torch.nn.functional as F


def compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta):
    cur_scores = ((token_level_scores * response_mask).sum(dim=1) * beta).sigmoid()
    cur_dpo_loss = torch.nn.functional.binary_cross_entropy(cur_scores, acc) # weakness: recognize current step as a positive sample when its trajectory is successful
    return cur_dpo_loss

# def compute_bt_loss_rm(
#     token_level_scores: torch.Tensor,   # (N, T)
#     acc: torch.Tensor,                  # (N,) in {0,1}
#     uid: torch.Tensor,                  # (N,) int64 tensor
#     response_mask: torch.Tensor,        # (N, T)
#     beta: float = 0.05,
# ):
#     device = token_level_scores.device
#     response_mask = response_mask.float()

#     q_seq = (token_level_scores * response_mask).sum(dim=-1)  # (N,)
#     uid = uid.to(device)

#     total_loss = torch.zeros((), device=device)
#     total_pairs = 0

#     for u in torch.unique(uid):
#         idxs = (uid == u).nonzero(as_tuple=False).squeeze(-1)  # (K,)
#         q_g = q_seq[idxs]
#         acc_g = acc[idxs].float()

#         q_pos = q_g[acc_g > 0.5]
#         q_neg = q_g[acc_g <= 0.5]

#         n = q_pos.numel()
#         m = q_neg.numel()
#         if n == 0 or m == 0:
#             continue

#         diff = q_pos.unsqueeze(1) - q_neg.unsqueeze(0)  # (n, m)
#         pair_loss = F.softplus(-beta * diff)

#         total_loss = total_loss + pair_loss.sum()
#         total_pairs += n * m

#     if total_pairs == 0:
#         return (q_seq.sum() * 0.0), 0   # on-graph zero

#     return total_loss / total_pairs, total_pairs

def compute_bt_loss_rm(token_level_scores, acc, uid, response_mask, beta=0.05, lam=1e-3):
    device = token_level_scores.device
    response_mask = response_mask.to(device=device, dtype=token_level_scores.dtype)
    acc = acc.to(device=device, dtype=token_level_scores.dtype)
    uid = uid.to(device=device)

    # length-normalized sequence score
    lens = response_mask.sum(dim=-1).clamp_min(1.0)
    q_seq = (token_level_scores * response_mask).sum(dim=-1) / lens

    uid_losses = []
    for u in torch.unique(uid):
        idxs = (uid == u).nonzero(as_tuple=False).squeeze(-1)
        q_g = q_seq[idxs]
        a_g = acc[idxs]

        # center per-uid to remove offset freedom
        q_g = q_g - q_g.mean()

        q_pos = q_g[a_g > 0.5]
        q_neg = q_g[a_g <= 0.5]
        if q_pos.numel() == 0 or q_neg.numel() == 0:
            continue

        diff = q_pos[:, None] - q_neg[None, :]
        uid_losses.append(F.softplus(-beta * diff).mean())

    if len(uid_losses) == 0:
        return (q_seq.sum() * 0.0), 0

    bt_loss = torch.stack(uid_losses).mean()

    # pin reward scale (optional but helpful)
    reg = lam * (q_seq ** 2).mean()
    return bt_loss + reg, len(uid_losses)

# def compute_irl_loss_rm(
#     token_level_scores: torch.Tensor,   # (N, T)
#     acc: torch.Tensor,                  # (N,) in {0,1}
#     uid: torch.Tensor,                  # (N,) prompt id (int)
#     response_mask: torch.Tensor,        # (N, T) 0/1
#     beta: float = 0.05,               # optional multiplier
# ):
#     """
#     Maximum-likelihood IRL-style surrogate:
#       maximize  E[q | correct] - E[q | all]
#     so the loss is:
#       loss = E[q | all] - E[q | correct]

#     Ignores prompts where all correct or all incorrect.
#     Returns: (scalar_loss, num_prompts_used)
#     """
#     device = token_level_scores.device
#     response_mask = response_mask.float().to(device)
#     acc = acc.to(device).float()
#     uid = uid.to(device)

#     # cumulative reward per sample (rollout)
#     q_seq = (token_level_scores * response_mask).sum(dim=-1)  # (N,)

#     total_loss = torch.zeros((), device=device)
#     num_used = 0

#     for u in torch.unique(uid):
#         idxs = (uid == u).nonzero(as_tuple=False).squeeze(-1)  # (K,)
#         q_g = q_seq[idxs]          # (K,)
#         acc_g = acc[idxs]          # (K,)

#         pos_mask = acc_g > 0.5
#         neg_mask = ~pos_mask

#         n = pos_mask.sum().item()
#         m = neg_mask.sum().item()
#         if n == 0 or m == 0:
#             continue  # ignore: all correct or all incorrect

#         mean_pos = q_g[pos_mask].mean()
#         mean_all = q_g.mean()

#         # minimize (mean_all - mean_pos) == maximize (mean_pos - mean_all)
#         surrogate_loss_g = mean_all - mean_pos

#         total_loss = total_loss + surrogate_loss_g
#         num_used += 1

#     if num_used == 0:
#         # on-graph zero (keeps dtype/device/graph happy)
#         return (q_seq.sum() * 0.0), 0

#     return total_loss / num_used, num_used

def compute_irl_loss_rm(
    token_level_scores: torch.Tensor,   # (N, T)
    acc: torch.Tensor,                  # (N,) in {0,1} (or float in [0,1])
    uid: torch.Tensor,                  # (N,) prompt id (int64)
    response_mask: torch.Tensor,        # (N, T) 0/1
    beta: float = 1.0,                  # optional scale on q_seq (keeps same logic)
    length_norm: str = "mean",          # {"sum","mean"}: keep same logic on q_seq, just normalize
    group_weight: str = "uniform",      # {"uniform","by_group_size","by_pairs"}
    q_clip: float = 10.0,               # clip q_seq to prevent blow-up (None disables)
    l2_reg: float = 1e-4,               # L2 on token scores to control scale (0 disables)
):
    """
    Keeps your original high-level logic per uid:
        loss_g = mean_all(q_g) - mean_pos(q_g)

    Allowed stabilizers:
      - length normalization (sum vs mean over response tokens)
      - optional per-seq clipping (q_clip)
      - optional L2 regularization on token scores (l2_reg)
      - optional group weighting when averaging across uids (group_weight)
      - optional scaling of q_seq via beta (does NOT change the loss form)

    Returns: (scalar_loss, num_used)
    """
    device = token_level_scores.device
    dtype = token_level_scores.dtype

    response_mask = response_mask.to(device=device, dtype=dtype)
    acc = acc.to(device=device, dtype=dtype)
    uid = uid.to(device=device)

    # --------- sequence score (same idea, but optionally length-normalized) ----------
    # q_seq = sum_t score_t over response tokens  (or mean_t if length_norm="mean")
    token_scores_masked = token_level_scores * response_mask
    q_sum = token_scores_masked.sum(dim=-1)  # (N,)
    if length_norm == "sum":
        q_seq = q_sum
    elif length_norm == "mean":
        resp_len = response_mask.sum(dim=-1).clamp_min(1.0)
        q_seq = q_sum / resp_len
    else:
        raise ValueError(f"length_norm must be 'sum' or 'mean', got {length_norm}")

    # optional scale (keeps same loss form)
    if beta is not None and beta != 1.0:
        q_seq = q_seq / max(float(beta), 1e-8)

    # optional clipping to prevent runaway magnitudes
    if q_clip is not None:
        q_seq = q_seq.clamp(min=-float(q_clip), max=float(q_clip))

    # --------- optional scale control on token-level scores ----------
    # (does not change your mean_all - mean_pos logic; just prevents exploding scores)
    if l2_reg and l2_reg > 0:
        denom = response_mask.sum().clamp_min(1.0)
        reg = float(l2_reg) * (token_scores_masked.pow(2).sum() / denom)
    else:
        reg = torch.zeros((), device=device, dtype=dtype)

    # --------- per-uid loss: mean_all - mean_pos ----------
    total_loss = torch.zeros((), device=device, dtype=dtype)
    total_weight = torch.zeros((), device=device, dtype=dtype)
    num_used = 0

    for u in torch.unique(uid):
        idxs = (uid == u).nonzero(as_tuple=False).squeeze(-1)  # (K,)
        q_g = q_seq[idxs]   # (K,)
        acc_g = acc[idxs]   # (K,)

        pos_mask = acc_g > 0.5
        neg_mask = ~pos_mask

        n = int(pos_mask.sum().item())
        m = int(neg_mask.sum().item())
        if n == 0 or m == 0:
            continue

        mean_pos = q_g[pos_mask].mean()
        mean_all = q_g.mean()

        # YOUR ORIGINAL LOGIC
        loss_g = mean_all - mean_pos

        # weighting across groups (still the same per-group loss)
        if group_weight == "uniform":
            w = 1.0
        elif group_weight == "by_group_size":
            w = float(q_g.numel())         # K
        elif group_weight == "by_pairs":
            w = float(n * m)               # number of pos-neg pairs in the group
        else:
            raise ValueError(
                f"group_weight must be 'uniform', 'by_group_size', or 'by_pairs', got {group_weight}"
            )

        total_loss = total_loss + loss_g * w
        total_weight = total_weight + w
        num_used += 1

    if num_used == 0:
        # on-graph zero, but keep reg so RM doesn't drift
        return (q_seq.sum() * 0.0 + reg), 0

    return (total_loss / total_weight) + reg, num_used

def composite_rm_loss(
    token_level_scores: torch.Tensor,   # (N,T)
    acc: torch.Tensor,                  # (N,) {0,1}
    uid: torch.Tensor,                  # (N,)
    response_mask: torch.Tensor,        # (N,T) {0,1}
    beta: float = 0.05,
    lambda_ce: float = 1.0,
    lambda_bt: float = 0.2,
    lambda_irl: float = 0.2,
):
    device = token_level_scores.device
    response_mask = response_mask.to(device=device, dtype=token_level_scores.dtype)
    acc = acc.to(device=device).float()
    uid = uid.to(device=device)

    # unified, length-normalized q_seq
    len_resp = response_mask.sum(dim=-1).clamp_min(1.0)
    q_seq = (token_level_scores * response_mask).sum(dim=-1) / len_resp  # (N,)

    # (A) CE/DPO-style: BCE on logits (stable)
    loss_ce = F.binary_cross_entropy_with_logits(beta * q_seq, acc)

    # (B) BT pairwise per-uid (balanced)
    per_uid_bt = []
    bt_pairs = 0
    for u in torch.unique(uid):
        idxs = (uid == u).nonzero(as_tuple=False).squeeze(-1)
        q_g = q_seq[idxs]
        a_g = acc[idxs]
        q_pos = q_g[a_g > 0.5]
        q_neg = q_g[a_g <= 0.5]
        if q_pos.numel() == 0 or q_neg.numel() == 0:
            continue
        diff = q_pos[:, None] - q_neg[None, :]
        per_uid_bt.append(F.softplus(-beta * diff).mean())
        bt_pairs += q_pos.numel() * q_neg.numel()

    loss_bt = (torch.stack(per_uid_bt).mean() if len(per_uid_bt) else (q_seq.sum() * 0.0))

    # (C) IRL surrogate per-uid (balanced)
    per_uid_irl = []
    irl_used = 0
    for u in torch.unique(uid):
        idxs = (uid == u).nonzero(as_tuple=False).squeeze(-1)
        q_g = q_seq[idxs]
        a_g = acc[idxs]
        pos = a_g > 0.5
        neg = ~pos
        if pos.sum() == 0 or neg.sum() == 0:
            continue
        mean_pos = q_g[pos].mean()
        mean_all = q_g.mean()
        per_uid_irl.append(mean_all - mean_pos)
        irl_used += 1

    loss_irl = (torch.stack(per_uid_irl).mean() if len(per_uid_irl) else (q_seq.sum() * 0.0))

    loss = lambda_ce * loss_ce + lambda_bt * loss_bt + lambda_irl * loss_irl

    metrics = {
        "loss_total": loss.detach(),
        "loss_ce": loss_ce.detach(),
        "loss_bt": loss_bt.detach(),
        "loss_irl": loss_irl.detach(),
        "bt_pairs": torch.tensor(bt_pairs, device=device),
        "irl_used": torch.tensor(irl_used, device=device),
        "q_seq_mean": q_seq.detach().mean(),
        "q_seq_std": q_seq.detach().std(unbiased=False),
        "len_mean": len_resp.detach().mean(),
        "len_max": len_resp.detach().max(),
    }
    return loss, metrics


def compute_detach_dpo_loss_rm(token_level_scores, acc, Q_bc, acc_bc, response_mask, beta, bon_mode="none"):
    # we always assume that the BoN size equals n_samples
    # mode1: use acc as rm
    # mode2: use Q as rm
    cur_Q = (token_level_scores * response_mask).sum(dim=1) * beta
    other_Q = torch.zeros_like(cur_Q)
    for i in range(token_level_scores.shape[0]):
        Q_chosen = Q_bc[i][acc_bc[i] < acc[i]] if acc[i] > 0 else Q_bc[i][acc_bc[i] > acc[i]]
        if len(Q_chosen) > 0:
            other_Q[i] = Q_chosen.mean() * beta
        else:
            other_Q[i] = 0 # reduce to ce_loss but may cause numerical instability
    dpo_loss = -torch.log(torch.sigmoid((cur_Q - other_Q) * ((acc > 0).float() * 2 - 1)))
    if bon_mode == "none":
        dpo_loss = dpo_loss.mean()
    else: # higher weight for samples that outperform more references
        weight = torch.zeros_like(dpo_loss)
        n_samples = acc_bc.shape[1]
        if bon_mode == "bon_rm":
            for i in range(token_level_scores.shape[0]):
                weight[i] = n_samples * torch.pow((Q_bc[i] * beta <= cur_Q[i]).float().mean(), n_samples - 1)
        elif bon_mode == "bon_acc":
            for i in range(token_level_scores.shape[0]):
                weight[i] = n_samples * torch.pow((acc_bc[i] <= acc[i]).float().mean(), n_samples - 1)
        else:
            raise NotImplementedError
        dpo_loss = (dpo_loss * weight).sum()

    return dpo_loss

def compute_eto_loss_rm(data: DataProto, token_level_scores, beta):
    """
    Compute DPO loss using trajectory-level preferences with one-to-one pairing.
    Each successful trajectory is paired with exactly one unsuccessful trajectory,
    avoiding reuse of trajectories in subsequent comparisons.
    
    Args:
        data: DataProto containing batch information
        token_level_scores: Q-values for each step (batch_size, response_length)
        beta: Temperature parameter for DPO
    Returns:
        dpo_loss: Computed DPO loss
    """
    if not token_level_scores.requires_grad:
        token_level_scores = token_level_scores.requires_grad_(True)
    
    # Extract trajectory information
    traj_uids = data.non_tensor_batch['traj_uid']
    uid_batch = data.non_tensor_batch['uid']  # Environment group IDs
    episode_rewards = data.batch['acc']  # Trajectory-level success
    active_masks = data.non_tensor_batch['active_masks']
    
    # Step 1: Aggregate step-level Q-values to trajectory-level
    traj_q_values = {}  # Maps traj_uid -> trajectory Q-value
    traj_success = {}   # Maps traj_uid -> trajectory success (episode_rewards)
    traj_env_group = {} # Maps traj_uid -> environment group (uid)
    
    # Compute Q-values for each step
    prompt_ids = data.batch['prompts']
    prompt_length = prompt_ids.shape[-1]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, prompt_length:]
    step_q_values = (token_level_scores * response_mask).sum(dim=1) * beta
    
    for i in range(len(traj_uids)):
        if not active_masks[i]:  # Skip inactive steps
            continue
            
        traj_uid = traj_uids[i]
        env_group = uid_batch[i]
        
        if traj_uid not in traj_q_values:
            traj_q_values[traj_uid] = []
            traj_success[traj_uid] = episode_rewards[i]
            traj_env_group[traj_uid] = env_group
        
        traj_q_values[traj_uid].append(step_q_values[i])
    
    for traj_uid in traj_q_values:
        traj_q_values[traj_uid] = torch.stack(traj_q_values[traj_uid]).mean()
    
    # Step 2: Group trajectories by environment group
    env_groups = {}  # Maps env_group -> list of traj_uids
    for traj_uid, env_group in traj_env_group.items():
        if env_group not in env_groups:
            env_groups[env_group] = []
        env_groups[env_group].append(traj_uid)
    
    # Step 3: Construct one-to-one preference pairs within each environment group
    total_loss = 0.0
    total_pairs = 0
    
    for env_group, traj_list in env_groups.items():
        if len(traj_list) < 2:  
            continue
            
        # Separate successful and unsuccessful trajectories
        successful_trajs = [uid for uid in traj_list if traj_success[uid] > 0]
        unsuccessful_trajs = [uid for uid in traj_list if traj_success[uid] <= 0]
        
        # Handle edge cases
        if len(successful_trajs) == 0 or len(unsuccessful_trajs) == 0:
            continue  
        
        for i in range(len(successful_trajs)):
            for j in range(len(unsuccessful_trajs)):
                succ_traj = successful_trajs[i]
                unsucc_traj = unsuccessful_trajs[j]
                
                succ_q = traj_q_values[succ_traj]
                unsucc_q = traj_q_values[unsucc_traj]
                
                # DPO loss: successful trajectory should have higher Q-value
                pair_loss = -torch.log(torch.sigmoid(succ_q - unsucc_q))
                total_loss += pair_loss
                total_pairs += 1
    
    if total_pairs == 0:
        # No valid pairs found, return zero loss
        print("Warning: No valid trajectory pairs found for DPO training")
        return torch.tensor(0.0, device=token_level_scores.device, requires_grad=True)
    
    return total_loss / total_pairs

def compute_dpo_accuracy(token_level_scores, acc, response_mask, n_samples):
    if n_samples is None or n_samples < 2:
        return torch.tensor(0.5, device=token_level_scores.device)
    dpo_acc = []
    for start_id in range(0, token_level_scores.shape[0], n_samples):
        cur_scores = (token_level_scores[start_id : start_id + n_samples] * response_mask[start_id : start_id + n_samples]).sum(dim=1)

        def get_upper_triangle(tensor_x):
            diff_matrix = tensor_x.unsqueeze(1) - tensor_x.unsqueeze(0)
            upper_tri_indices = torch.triu(torch.ones_like(diff_matrix).bool(), diagonal=1)
            return diff_matrix[upper_tri_indices]

        cur_acc_diff = get_upper_triangle(acc[start_id : start_id + n_samples])  # in range [-1,1]
        cur_score_diff = get_upper_triangle(cur_scores)  # in R
        cur_score_prediction = (cur_score_diff > 0).float()  # in [0,1]
        if cur_acc_diff.abs().sum() == 0:
            cur_acc = torch.zeros_like(cur_score_prediction[0]) + 0.5
        else:
            cur_acc = (((cur_score_diff > 0) == (cur_acc_diff > 0)).float() * cur_acc_diff.abs()).sum() / cur_acc_diff.abs().sum()

        dpo_acc.append(cur_acc.unsqueeze(0))

    return torch.cat(dpo_acc, dim=0).mean()

def compute_dpo_abs_accuracy(token_level_scores, acc, response_mask, n_samples):
    return (torch.sign((token_level_scores * response_mask).sum(dim=-1)) == torch.sign(acc * 2 - 1)).float().mean()
