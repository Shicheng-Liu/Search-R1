import random

import torch
from collections import defaultdict
import verl
from verl import DataProto
import torch.nn.functional as F


def compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta):
    cur_scores = ((token_level_scores * response_mask).sum(dim=1) * beta).sigmoid()
    cur_dpo_loss = torch.nn.functional.binary_cross_entropy(cur_scores, acc) 
    return cur_dpo_loss


def _to_list(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return list(x)


def check_uid_grouping(uid, m, max_print=10):
    uid_list = _to_list(uid)

    N = len(uid_list)
    if N % m != 0:
        print(f"[UID CHECK FAILED] N={N} is not divisible by m={m}")
        return False

    B = N // m
    bad_groups = []

    for b in range(B):
        group = uid_list[b * m:(b + 1) * m]
        if len(set(group)) != 1:
            bad_groups.append((b, group))

    if bad_groups:
        print(f"[UID CHECK FAILED] {len(bad_groups)} / {B} groups are not correctly grouped.")
        for b, group in bad_groups[:max_print]:
            print(f"  group {b}: {group}")
        return False

    print(f"[UID CHECK PASSED] All {B} groups have exactly m={m} samples from the same uid.")
    return True


def compute_ce_hyper_loss_rm(
    token_level_scores,
    acc,
    response_mask,
    beta,
    m=8,
    uid=None,
    gamma=1.0,
    debug=False,
):
    """
    token_level_scores: [B*m, L], unscaled r_theta token scores
    acc:                [B*m], binary labels
    response_mask:      [B*m, L]
    beta:               code beta, direct CE uses beta * r_theta
    uid:                optional [B*m], prompt uid
    gamma:              discount factor. gamma=1 reproduces your current suffix-sum.
    """

    if token_level_scores.ndim != 2:
        raise ValueError(f"token_level_scores must be [B*m, L], got {token_level_scores.shape}")

    if response_mask.shape != token_level_scores.shape:
        raise ValueError(
            f"response_mask must match token_level_scores shape, got "
            f"{response_mask.shape} vs {token_level_scores.shape}"
        )

    if acc.ndim != 1 or acc.shape[0] != token_level_scores.shape[0]:
        raise ValueError(
            f"acc must be [B*m], got {acc.shape}, expected first dim {token_level_scores.shape[0]}"
        )

    N, L = token_level_scores.shape
    if N % m != 0:
        raise ValueError(f"Batch size N={N} must be divisible by m={m}")

    B = N // m
    dtype = token_level_scores.dtype
    device = token_level_scores.device

    acc = acc.to(device=device, dtype=dtype)
    response_mask = response_mask.to(device=device, dtype=dtype)

    if uid is not None and debug:
        check_uid_grouping(uid, m)

    # ------------------------------------------------------------
    # Direct CE term
    # ------------------------------------------------------------
    traj_raw_scores = (token_level_scores * response_mask).sum(dim=1)       # [B*m]
    traj_logits = traj_raw_scores * beta                                    # [B*m]

    ce_per_traj = F.binary_cross_entropy_with_logits(
        traj_logits, acc, reduction="none"
    )                                                                       # [B*m]

    direct_ce_loss = ce_per_traj.mean()

    # ------------------------------------------------------------
    # Indirect term
    # Because implemented reward is beta * r_theta and paper has 1/beta,
    # beta cancels. So use unscaled token_level_scores here.
    # ------------------------------------------------------------
    # masked_scores = token_level_scores * response_mask                      # [B*m, L]
    # discount = gamma ** torch.arange(L, device=device, dtype=dtype)
    # if gamma != 1.0:
    #     # token index i corresponds approximately to discount gamma^i
    #     discount = gamma ** torch.arange(L, device=device, dtype=dtype)      # [L]
    #     masked_scores_for_suffix = masked_scores * discount.unsqueeze(0)
    # else:
    #     masked_scores_for_suffix = masked_scores

    # suffix_scores = torch.flip(
    #     torch.cumsum(torch.flip(masked_scores_for_suffix, dims=[1]), dim=1),
    #     dims=[1],
    # )                                                                       # [B*m, L]

    # suffix_scores_grouped = suffix_scores.view(B, m, L)                     # [B, m, L]
    ce_per_traj_grouped = ce_per_traj.view(B, m)                            # [B, m]
    acc_grouped = acc.view(B, m)                                             # [B, m]

    # suffix_baseline = suffix_scores_grouped.mean(dim=1, keepdim=True)       # [B, 1, L]
    # centered_suffix = suffix_scores_grouped - suffix_baseline               # [B, m, L]

    # implicit_scalar = centered_suffix.sum(dim=2)                            # [B, m]
    # implicit_ce_loss = (ce_per_traj_grouped.detach() * implicit_scalar).mean()

    # total_loss = direct_ce_loss + implicit_ce_loss

    lambda_indirect = 1e-2
    traj_scores_grouped = (token_level_scores * response_mask).sum(dim=1).view(B, m)
    traj_baseline = traj_scores_grouped.mean(dim=1, keepdim=True)
    implicit_scalar = traj_scores_grouped - traj_baseline

    implicit_ce_loss = (ce_per_traj_grouped.detach() * implicit_scalar).mean()

    total_loss = direct_ce_loss + lambda_indirect * implicit_ce_loss

    # ------------------------------------------------------------
    # Debug prints
    # ------------------------------------------------------------
    if debug:
        with torch.no_grad():
            pred = torch.sigmoid(traj_logits)

            pos_mask = acc == 1
            neg_mask = acc == 0
            valid_mask = response_mask.bool()

            print("\n========== CE HYPER LOSS DEBUG ==========")
            print(f"N={N}, B={B}, m={m}, L={L}, beta={beta}, gamma={gamma}")
            print(f"token_level_scores.requires_grad: {token_level_scores.requires_grad}")
            print(f"response length mean/min/max: "
                  f"{response_mask.sum(dim=1).float().mean().item():.3f} / "
                  f"{response_mask.sum(dim=1).min().item():.0f} / "
                  f"{response_mask.sum(dim=1).max().item():.0f}")

            print("\n--- Label / prediction stats ---")
            print(f"acc mean: {acc.float().mean().item():.6f}")
            print(f"num pos: {pos_mask.sum().item()}, num neg: {neg_mask.sum().item()}")
            print(f"pred mean/std/min/max: "
                  f"{pred.mean().item():.6f} / "
                  f"{pred.std().item():.6f} / "
                  f"{pred.min().item():.6f} / "
                  f"{pred.max().item():.6f}")

            print("\n--- Trajectory score stats ---")
            print(f"traj_raw_scores mean/std/min/max: "
                  f"{traj_raw_scores.mean().item():.6f} / "
                  f"{traj_raw_scores.std().item():.6f} / "
                  f"{traj_raw_scores.min().item():.6f} / "
                  f"{traj_raw_scores.max().item():.6f}")
            print(f"traj_logits mean/std/min/max: "
                  f"{traj_logits.mean().item():.6f} / "
                  f"{traj_logits.std().item():.6f} / "
                  f"{traj_logits.min().item():.6f} / "
                  f"{traj_logits.max().item():.6f}")

            print("\n--- CE stats ---")
            print(f"direct_ce_loss: {direct_ce_loss.item():.6f}")
            print(f"ce mean/std/min/max: "
                  f"{ce_per_traj.mean().item():.6f} / "
                  f"{ce_per_traj.std().item():.6f} / "
                  f"{ce_per_traj.min().item():.6f} / "
                  f"{ce_per_traj.max().item():.6f}")

            if pos_mask.any():
                print(f"ce positive mean: {ce_per_traj[pos_mask].mean().item():.6f}")
                print(f"pred positive mean: {pred[pos_mask].mean().item():.6f}")
                print(f"raw score positive mean: {traj_raw_scores[pos_mask].mean().item():.6f}")

            if neg_mask.any():
                print(f"ce negative mean: {ce_per_traj[neg_mask].mean().item():.6f}")
                print(f"pred negative mean: {pred[neg_mask].mean().item():.6f}")
                print(f"raw score negative mean: {traj_raw_scores[neg_mask].mean().item():.6f}")

            print("\n--- Implicit term stats ---")
            print(f"implicit_scalar mean/std/min/max: "
                  f"{implicit_scalar.mean().item():.6f} / "
                  f"{implicit_scalar.std().item():.6f} / "
                  f"{implicit_scalar.min().item():.6f} / "
                  f"{implicit_scalar.max().item():.6f}")
            print("implicit_ce_loss (raw):", implicit_ce_loss.item())
            print("implicit_ce_loss (scaled):", (lambda_indirect * implicit_ce_loss).item())
            print(f"total_loss: {total_loss.item():.6f}")

            print("\n--- Group-level stats ---")
            group_pos_counts = acc_grouped.sum(dim=1)
            print(f"group positive count mean/min/max: "
                  f"{group_pos_counts.float().mean().item():.3f} / "
                  f"{group_pos_counts.min().item():.0f} / "
                  f"{group_pos_counts.max().item():.0f}")
            print(f"groups all positive: {(group_pos_counts == m).sum().item()} / {B}")
            print(f"groups all negative: {(group_pos_counts == 0).sum().item()} / {B}")
            print("q abs mean:", token_level_scores.abs().mean().item())
            print("q std:", token_level_scores.std().item())
            print("=========================================\n")

    # ------------------------------------------------------------
    # Gradient diagnostics
    # ------------------------------------------------------------
    if debug and token_level_scores.requires_grad:
        direct_grad = torch.autograd.grad(
            direct_ce_loss,
            token_level_scores,
            retain_graph=True,
            allow_unused=True,
        )[0]

        implicit_grad = torch.autograd.grad(
            implicit_ce_loss,
            token_level_scores,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if direct_grad is None:
            print("[GRAD DEBUG] direct_grad is None")
        if implicit_grad is None:
            print("[GRAD DEBUG] implicit_grad is None")

        if direct_grad is not None and implicit_grad is not None:
            with torch.no_grad():
                mask = response_mask.bool()

                dg = direct_grad[mask].flatten()
                ig = implicit_grad[mask].flatten()

                direct_norm = dg.norm()
                implicit_norm = ig.norm()

                if direct_norm.item() > 0 and implicit_norm.item() > 0:
                    cosine = F.cosine_similarity(dg, ig, dim=0)
                else:
                    cosine = torch.tensor(float("nan"), device=device)

                print("\n========== GRADIENT DEBUG ==========")
                print(f"direct_grad norm: {direct_norm.item():.6e}")
                print(f"implicit_grad norm (raw): {implicit_norm.item():.6e}")
                print(f"implicit_grad norm (scaled): {(lambda_indirect * implicit_norm).item():.6e}")
                print(f"implicit/direct norm ratio: "
                      f"{(lambda_indirect *implicit_norm / (direct_norm + 1e-12)).item():.6e}")
                print(f"grad cosine direct vs implicit: {cosine.item():.6f}")

                total_grad = direct_grad + implicit_grad
                tg = total_grad[mask].flatten()
                print(f"total_grad norm: {tg.norm().item():.6e}")

                pos_token_mask = (acc[:, None] == 1) & mask
                neg_token_mask = (acc[:, None] == 0) & mask

                if pos_token_mask.any():
                    print(f"direct grad mean on positive tokens: "
                          f"{direct_grad[pos_token_mask].mean().item():.6e}")
                    print(f"implicit grad mean on positive tokens: "
                          f"{implicit_grad[pos_token_mask].mean().item():.6e}")
                    print(f"total grad mean on positive tokens: "
                          f"{total_grad[pos_token_mask].mean().item():.6e}")

                if neg_token_mask.any():
                    print(f"direct grad mean on negative tokens: "
                          f"{direct_grad[neg_token_mask].mean().item():.6e}")
                    print(f"implicit grad mean on negative tokens: "
                          f"{implicit_grad[neg_token_mask].mean().item():.6e}")
                    print(f"total grad mean on negative tokens: "
                          f"{total_grad[neg_token_mask].mean().item():.6e}")

                print("====================================\n")

    return total_loss

# def compute_ce_hyper_loss_rm(token_level_scores, acc, response_mask, beta, m=8):
#     """
#     token_level_scores: [B*m, L]
#     acc:                [B*m]
#     response_mask:      [B*m, L]
#     m:                  number of sampled trajectories per prompt
#     """
#     if token_level_scores.ndim != 2:
#         raise ValueError(f"token_level_scores must be [B*m, L], got {token_level_scores.shape}")
#     if response_mask.shape != token_level_scores.shape:
#         raise ValueError(
#             f"response_mask must match token_level_scores shape, got "
#             f"{response_mask.shape} vs {token_level_scores.shape}"
#         )
#     if acc.ndim != 1 or acc.shape[0] != token_level_scores.shape[0]:
#         raise ValueError(
#             f"acc must be [B*m], got {acc.shape}, expected first dim {token_level_scores.shape[0]}"
#         )

#     N, L = token_level_scores.shape
#     if N % m != 0:
#         raise ValueError(f"Batch size {N} must be divisible by m={m}")

#     B = N // m
#     dtype = token_level_scores.dtype

#     acc = acc.to(dtype=dtype)
#     response_mask = response_mask.to(dtype=dtype)

#     # ------------------------------------------------------------
#     # Direct CE term
#     # ------------------------------------------------------------
#     traj_logits = (token_level_scores * response_mask).sum(dim=1) * beta   # [B*m]
#     ce_per_traj = F.binary_cross_entropy_with_logits(
#         traj_logits, acc, reduction="none"
#     )  # [B*m]
#     direct_ce_loss = ce_per_traj.mean()

#     # ------------------------------------------------------------
#     # Implicit term surrogate
#     # Use the same beta-scaled reward pieces for consistency
#     # ------------------------------------------------------------
#     scaled_scores = token_level_scores
#     masked_scaled_scores = scaled_scores * response_mask                    # [B*m, L]

#     # suffix sums over token dimension
#     suffix_scores = torch.flip(
#         torch.cumsum(torch.flip(masked_scaled_scores, dims=[1]), dim=1),
#         dims=[1],
#     )  # [B*m, L]

#     # group by prompt
#     suffix_scores = suffix_scores.view(B, m, L)                            # [B, m, L]
#     ce_per_traj_grouped = ce_per_traj.view(B, m)                           # [B, m]

#     # within-prompt baseline
#     suffix_baseline = suffix_scores.mean(dim=1, keepdim=True)              # [B, 1, L]

#     # scalar surrogate
#     implicit_scalar = (suffix_scores - suffix_baseline).sum(dim=2)         # [B, m]

#     # detach CE multiplier
#     implicit_ce_loss = (ce_per_traj_grouped.detach() * implicit_scalar).mean()

#     total_loss = direct_ce_loss + implicit_ce_loss
#     return total_loss

# def compute_ce_dpo_loss_rm(token_level_scores, acc, response_mask, beta):
#     # sum of token rewards
#     sum_scores = (token_level_scores * response_mask).sum(dim=1)

#     # number of valid tokens
#     lengths = response_mask.sum(dim=1).clamp(min=1)

#     # average token reward
#     avg_scores = sum_scores / lengths

#     # sigmoid prediction
#     cur_scores = torch.sigmoid(beta * avg_scores)

#     # BCE loss
#     cur_dpo_loss = F.binary_cross_entropy(cur_scores, acc)

#     return cur_dpo_loss

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


##############stable version#######################
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

def compute_bt_hyper_loss_rm(
    token_level_scores,
    acc,
    uid,
    response_mask,
    beta=0.05,
    lam=1e-3,
):
    """
    Full BT / DPO-style hyper-loss:
      1) direct pairwise BT loss
      2) implicit dependency term via surrogate loss

    Args:
        token_level_scores: [N, L]
        acc:                [N]       binary success labels
        uid:                [N]       prompt/group ids
        response_mask:      [N, L]
        beta:               scaling used in the direct pairwise loss and implicit surrogate
        lam:                L2 regularizer coefficient

    Returns:
        total_loss: scalar
        num_valid_uid_groups: int
    """
    device = token_level_scores.device
    dtype = token_level_scores.dtype

    response_mask = response_mask.to(device=device, dtype=dtype)
    acc = acc.to(device=device, dtype=dtype)
    uid = uid.to(device=device)

    # ------------------------------------------------------------
    # Per-trajectory sequence score
    # ------------------------------------------------------------
    lens = response_mask.sum(dim=-1).clamp_min(1.0)  # [N]
    q_seq = (token_level_scores * response_mask).sum(dim=-1) / lens  # [N]

    # For implicit surrogate, use the same beta-scaled reward convention
    scaled_token_scores = beta * token_level_scores
    scaled_q_seq = (scaled_token_scores * response_mask).sum(dim=-1) / lens  # [N]

    uid_direct_losses = []
    uid_implicit_losses = []

    unique_uid = torch.unique(uid)

    for u in unique_uid:
        idxs = (uid == u).nonzero(as_tuple=False).squeeze(-1)

        q_g = q_seq[idxs]                 # [g]
        q_g_scaled = scaled_q_seq[idxs]   # [g]
        a_g = acc[idxs]                   # [g]

        # center per-uid to remove offset freedom, matching your direct code
        q_g = q_g - q_g.mean()
        q_g_scaled = q_g_scaled - q_g_scaled.mean()

        pos_mask = a_g > 0.5
        neg_mask = ~pos_mask

        q_pos = q_g[pos_mask]                   # [P]
        q_neg = q_g[neg_mask]                   # [Q]
        q_pos_scaled = q_g_scaled[pos_mask]     # [P]
        q_neg_scaled = q_g_scaled[neg_mask]     # [Q]

        if q_pos.numel() == 0 or q_neg.numel() == 0:
            continue

        # --------------------------------------------------------
        # Direct BT / DPO loss for all positive-negative pairs
        # --------------------------------------------------------
        diff = q_pos[:, None] - q_neg[None, :]               # [P, Q]
        pair_loss = F.softplus(-beta * diff)                 # [P, Q]
        uid_direct_losses.append(pair_loss.mean())

        # --------------------------------------------------------
        # Implicit surrogate term
        #
        # Appendix DPO correction structure:
        #   (+ trajectory term) + (- trajectory term) - 2 * group baseline
        #
        # We build a scalar surrogate with the same structure.
        # Since your direct loss uses sequence scores q_seq, we use the
        # beta-scaled sequence scores consistently here as requested.
        # --------------------------------------------------------
        group_mean_scaled = q_g_scaled.mean()                # scalar

        # scalar surrogate for each positive-negative pair
        # shape [P, Q]
        implicit_scalar_pairs = (
            q_pos_scaled[:, None]
            + q_neg_scaled[None, :]
            - 2.0 * group_mean_scaled
        )

        # detach pairwise direct loss coefficient
        uid_implicit_losses.append(
            (pair_loss.detach() * implicit_scalar_pairs).mean()
        )

    if len(uid_direct_losses) == 0:
        zero = q_seq.sum() * 0.0
        return zero, 0

    direct_bt_loss = torch.stack(uid_direct_losses).mean()
    implicit_bt_loss = torch.stack(uid_implicit_losses).mean()

    # same regularizer as your original code
    reg = lam * (q_seq ** 2).mean()
    lambda_indirect = 1e-2
    total_loss = direct_bt_loss + lambda_indirect * implicit_bt_loss + reg
    return total_loss, len(uid_direct_losses)

##############replay buffer version#############################
# def compute_bt_loss_rm(
#     token_level_scores,
#     acc,
#     uid,
#     response_mask,
#     beta=0.05,
#     lam_seq=1e-3,
#     lam_token=1e-4,
#     eps=1e-4,
# ):
#     device = token_level_scores.device
#     response_mask = response_mask.to(device=device, dtype=token_level_scores.dtype)
#     acc = acc.to(device=device, dtype=token_level_scores.dtype)
#     uid = uid.to(device=device)

#     # length-normalized sequence score
#     lens = response_mask.sum(dim=-1).clamp_min(1.0)
#     q_seq = (token_level_scores * response_mask).sum(dim=-1) / lens

#     uid_losses = []
#     valid_uid_count = 0

#     for u in torch.unique(uid):
#         idxs = (uid == u).nonzero(as_tuple=False).squeeze(-1)
#         q_g = q_seq[idxs]
#         a_g = acc[idxs]

#         q_pos = q_g[a_g > 0.5]
#         q_neg = q_g[a_g <= 0.5]
#         if q_pos.numel() == 0 or q_neg.numel() == 0:
#             continue

#         # stabilize per-uid score scale
#         q_g = (q_g - q_g.mean()) / q_g.std(unbiased=False).clamp_min(eps)

#         # recompute pos/neg after normalization
#         q_pos = q_g[a_g > 0.5]
#         q_neg = q_g[a_g <= 0.5]

#         diff = q_pos[:, None] - q_neg[None, :]
#         logits = (beta * diff).clamp(min=-20.0, max=20.0)
#         uid_losses.append(F.softplus(-logits).mean())
#         valid_uid_count += 1

#     if valid_uid_count == 0:
#         return q_seq.sum() * 0.0, 0

#     bt_loss = torch.stack(uid_losses).mean()

#     reg_seq = lam_seq * (q_seq ** 2).mean()

#     masked_scores = token_level_scores * response_mask
#     reg_token = lam_token * (masked_scores ** 2).sum() / response_mask.sum().clamp_min(1.0)

#     loss = bt_loss + reg_seq + reg_token
#     return loss, valid_uid_count

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


##############stable version#######################
# def compute_irl_loss_rm(
#     token_level_scores: torch.Tensor,   # (N, T)
#     acc: torch.Tensor,                  # (N,) in {0,1} (or float in [0,1])
#     uid: torch.Tensor,                  # (N,) prompt id (int64)
#     response_mask: torch.Tensor,        # (N, T) 0/1
#     beta: float = 1.0,                  # optional scale on q_seq (keeps same logic)
#     length_norm: str = "mean",          # {"sum","mean"}: keep same logic on q_seq, just normalize
#     group_weight: str = "uniform",      # {"uniform","by_group_size","by_pairs"}
#     q_clip: float = 10.0,               # clip q_seq to prevent blow-up (None disables)
#     l2_reg: float = 1e-4,               # L2 on token scores to control scale (0 disables)
# ):
#     """
#     Keeps your original high-level logic per uid:
#         loss_g = mean_all(q_g) - mean_pos(q_g)

#     Allowed stabilizers:
#       - length normalization (sum vs mean over response tokens)
#       - optional per-seq clipping (q_clip)
#       - optional L2 regularization on token scores (l2_reg)
#       - optional group weighting when averaging across uids (group_weight)
#       - optional scaling of q_seq via beta (does NOT change the loss form)

#     Returns: (scalar_loss, num_used)
#     """
#     device = token_level_scores.device
#     dtype = token_level_scores.dtype

#     response_mask = response_mask.to(device=device, dtype=dtype)
#     acc = acc.to(device=device, dtype=dtype)
#     uid = uid.to(device=device)

#     # --------- sequence score (same idea, but optionally length-normalized) ----------
#     # q_seq = sum_t score_t over response tokens  (or mean_t if length_norm="mean")
#     token_scores_masked = token_level_scores * response_mask
#     q_sum = token_scores_masked.sum(dim=-1)  # (N,)
#     if length_norm == "sum":
#         q_seq = q_sum
#     elif length_norm == "mean":
#         resp_len = response_mask.sum(dim=-1).clamp_min(1.0)
#         q_seq = q_sum / resp_len
#     else:
#         raise ValueError(f"length_norm must be 'sum' or 'mean', got {length_norm}")

#     # optional scale (keeps same loss form)
#     if beta is not None and beta != 1.0:
#         q_seq = q_seq / max(float(beta), 1e-8)

#     # optional clipping to prevent runaway magnitudes
#     if q_clip is not None:
#         q_seq = q_seq.clamp(min=-float(q_clip), max=float(q_clip))

#     # --------- optional scale control on token-level scores ----------
#     # (does not change your mean_all - mean_pos logic; just prevents exploding scores)
#     if l2_reg and l2_reg > 0:
#         denom = response_mask.sum().clamp_min(1.0)
#         reg = float(l2_reg) * (token_scores_masked.pow(2).sum() / denom)
#     else:
#         reg = torch.zeros((), device=device, dtype=dtype)

#     # --------- per-uid loss: mean_all - mean_pos ----------
#     total_loss = torch.zeros((), device=device, dtype=dtype)
#     total_weight = torch.zeros((), device=device, dtype=dtype)
#     num_used = 0

#     for u in torch.unique(uid):
#         idxs = (uid == u).nonzero(as_tuple=False).squeeze(-1)  # (K,)
#         q_g = q_seq[idxs]   # (K,)
#         acc_g = acc[idxs]   # (K,)

#         pos_mask = acc_g > 0.5
#         neg_mask = ~pos_mask

#         n = int(pos_mask.sum().item())
#         m = int(neg_mask.sum().item())
#         if n == 0 or m == 0:
#             continue

#         mean_pos = q_g[pos_mask].mean()
#         mean_all = q_g.mean()

#         # YOUR ORIGINAL LOGIC
#         loss_g = mean_all - mean_pos

#         # weighting across groups (still the same per-group loss)
#         if group_weight == "uniform":
#             w = 1.0
#         elif group_weight == "by_group_size":
#             w = float(q_g.numel())         # K
#         elif group_weight == "by_pairs":
#             w = float(n * m)               # number of pos-neg pairs in the group
#         else:
#             raise ValueError(
#                 f"group_weight must be 'uniform', 'by_group_size', or 'by_pairs', got {group_weight}"
#             )

#         total_loss = total_loss + loss_g * w
#         total_weight = total_weight + w
#         num_used += 1

#     if num_used == 0:
#         # on-graph zero, but keep reg so RM doesn't drift
#         return (q_seq.sum() * 0.0 + reg), 0

#     return (total_loss / total_weight) + reg, num_used

###############################replay buffer version###################################
def compute_irl_loss_rm(
    token_level_scores: torch.Tensor,
    acc: torch.Tensor,
    uid: torch.Tensor,
    response_mask: torch.Tensor,
    beta: float = 1.0,
    length_norm: str = "mean",
    group_weight: str = "uniform",
    l2_reg: float = 1e-4,
    eps: float = 1e-8,
):
    device = token_level_scores.device
    dtype = token_level_scores.dtype

    response_mask = response_mask.to(device=device, dtype=dtype)
    acc = acc.to(device=device, dtype=dtype)
    uid = uid.to(device=device)

    token_scores_masked = token_level_scores * response_mask
    q_sum = token_scores_masked.sum(dim=-1)

    if length_norm == "sum":
        q_seq = q_sum
    elif length_norm == "mean":
        resp_len = response_mask.sum(dim=-1).clamp_min(1.0)
        q_seq = q_sum / resp_len
    else:
        raise ValueError(f"length_norm must be 'sum' or 'mean', got {length_norm}")

    # optional regularization on token scores
    if l2_reg > 0:
        denom = response_mask.sum().clamp_min(1.0)
        reg = l2_reg * (token_scores_masked.pow(2).sum() / denom)
    else:
        reg = torch.zeros((), device=device, dtype=dtype)

    total_loss = torch.zeros((), device=device, dtype=dtype)
    total_weight = torch.zeros((), device=device, dtype=dtype)
    num_used = 0

    for u in torch.unique(uid):
        idxs = (uid == u).nonzero(as_tuple=False).squeeze(-1)
        q_g = q_seq[idxs]
        acc_g = acc[idxs]

        pos_mask = acc_g > 0.5
        neg_mask = ~pos_mask

        n = int(pos_mask.sum().item())
        m = int(neg_mask.sum().item())
        if n == 0 or m == 0:
            continue

        mean_pos = q_g[pos_mask].mean()
        mean_neg = q_g[neg_mask].mean()

        gap = mean_pos - mean_neg
        loss_g = F.softplus(-beta * gap)

        if group_weight == "uniform":
            w = 1.0
        elif group_weight == "by_group_size":
            w = float(q_g.numel())
        elif group_weight == "by_pairs":
            w = float(n * m)
        else:
            raise ValueError(
                f"group_weight must be 'uniform', 'by_group_size', or 'by_pairs', got {group_weight}"
            )

        total_loss = total_loss + loss_g * w
        total_weight = total_weight + w
        num_used += 1

    if num_used == 0:
        return q_seq.sum() * 0.0 + reg, 0

    return total_loss / total_weight + reg, num_used

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
