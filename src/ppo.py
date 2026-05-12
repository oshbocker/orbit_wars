from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from .policy import PolicyOutput


@dataclass(slots=True)
class SampledAction:
    target_index: torch.Tensor     # [B] — 0=NoOp, 1..T are targets
    fraction_bin: torch.Tensor     # [B] — 0..num_fractions-1
    log_prob: torch.Tensor         # [B]
    entropy: torch.Tensor          # [B]


@dataclass(slots=True)
class TransitionBatch:
    # Features
    global_features: torch.Tensor      # [N, GLOBAL_DIM]
    source_scalars: torch.Tensor       # [N, SOURCE_SCALAR_DIM]
    source_positions: torch.Tensor     # [N, 2]
    knn_scalars: torch.Tensor          # [N, K, KNN_SCALAR_DIM]
    knn_positions: torch.Tensor        # [N, K, 2]
    target_scalars: torch.Tensor       # [N, T, TARGET_SCALAR_DIM]
    target_positions: torch.Tensor     # [N, T, 2]
    target_mask: torch.Tensor          # [N, T+2]
    # Actions
    target_index: torch.Tensor         # [N]
    fraction_bin: torch.Tensor         # [N]
    log_prob: torch.Tensor             # [N]
    # Returns
    returns: torch.Tensor              # [N]
    advantages: torch.Tensor           # [N]
    values: torch.Tensor               # [N] — old value predictions for value clipping


def _safe_target_logits(target_logits: torch.Tensor) -> torch.Tensor:
    """Ensure at least one valid logit per row."""
    invalid_rows = ~torch.isfinite(target_logits).any(dim=-1)
    if not invalid_rows.any():
        return target_logits
    safe = target_logits.clone()
    safe[invalid_rows, 0] = 0.0  # fallback to NoOp
    return safe


def sample_actions(outputs: PolicyOutput, deterministic: bool = False) -> SampledAction:
    """Sample target + fraction from policy outputs."""
    B = outputs.target_logits.shape[0]
    target_logits = _safe_target_logits(outputs.target_logits)  # [B, 1+T]

    # Sample target
    target_dist = Categorical(logits=target_logits)
    if deterministic:
        target_index = target_logits.argmax(dim=-1)  # [B]
    else:
        target_index = target_dist.sample()  # [B]
    target_log_prob = target_dist.log_prob(target_index)  # [B]
    target_entropy = target_dist.entropy()  # [B]

    # Sample fraction for non-NoOp selections
    # target_index=0 means NoOp; target_index=k (k>=1) maps to target k-1
    is_noop = target_index == 0  # [B]

    # Gather fraction logits for selected targets
    # fraction_logits: [B, T, num_fractions]
    # For target_index k>=1, we want fraction_logits[:, k-1, :]
    frac_idx = (target_index - 1).clamp(min=0)  # [B]
    frac_logits_selected = outputs.fraction_logits[
        torch.arange(B, device=outputs.fraction_logits.device), frac_idx
    ]  # [B, num_fractions]

    frac_dist = Categorical(logits=frac_logits_selected)
    if deterministic:
        fraction_bin = frac_logits_selected.argmax(dim=-1)  # [B]
    else:
        fraction_bin = frac_dist.sample()  # [B]
    frac_log_prob = frac_dist.log_prob(fraction_bin)  # [B]
    frac_entropy = frac_dist.entropy()  # [B]

    # Zero out fraction log_prob and entropy for NoOp
    frac_log_prob = frac_log_prob.masked_fill(is_noop, 0.0)
    frac_entropy = frac_entropy.masked_fill(is_noop, 0.0)

    total_log_prob = target_log_prob + frac_log_prob
    total_entropy = target_entropy + frac_entropy

    return SampledAction(
        target_index=target_index,
        fraction_bin=fraction_bin,
        log_prob=total_log_prob,
        entropy=total_entropy,
    )


def action_log_prob_and_entropy(
    outputs: PolicyOutput,
    target_index: torch.Tensor,
    fraction_bin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute log_prob and entropy for stored actions (used in PPO update)."""
    B = outputs.target_logits.shape[0]
    target_logits = _safe_target_logits(outputs.target_logits)

    # Target distribution
    target_dist = Categorical(logits=target_logits)
    target_log_prob = target_dist.log_prob(target_index)
    target_entropy = target_dist.entropy()

    # Fraction distribution
    is_noop = target_index == 0
    frac_idx = (target_index - 1).clamp(min=0)
    frac_logits_selected = outputs.fraction_logits[
        torch.arange(B, device=outputs.fraction_logits.device), frac_idx
    ]
    frac_dist = Categorical(logits=frac_logits_selected)
    frac_log_prob = frac_dist.log_prob(fraction_bin).masked_fill(is_noop, 0.0)
    frac_entropy = frac_dist.entropy().masked_fill(is_noop, 0.0)

    return target_log_prob + frac_log_prob, target_entropy + frac_entropy


def ppo_update(
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: TransitionBatch,
    *,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    epochs: int,
    minibatch_size: int,
    device: torch.device,
    demo_buffer: object | None = None,
    imitation_coef: float = 0.0,
) -> dict[str, float]:
    N = batch.global_features.shape[0]
    if N < 16:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                "imitation_loss": 0.0}

    # Move to device
    gf = batch.global_features.to(device)
    ss = batch.source_scalars.to(device)
    sp = batch.source_positions.to(device)
    ks = batch.knn_scalars.to(device)
    kp = batch.knn_positions.to(device)
    ts = batch.target_scalars.to(device)
    tp = batch.target_positions.to(device)
    tm = batch.target_mask.to(device).bool()
    old_log_prob = batch.log_prob.to(device)
    target_idx = batch.target_index.to(device)
    frac_bin = batch.fraction_bin.to(device)
    returns = batch.returns.to(device)
    advantages = batch.advantages.to(device)
    old_values = batch.values.to(device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    use_imitation = demo_buffer is not None and imitation_coef > 0.0

    minibatch_size = min(N, max(1, minibatch_size))
    metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
               "imitation_loss": 0.0}
    updates = 0

    for _ in range(epochs):
        order = torch.randperm(N, device=device)
        for start in range(0, N, minibatch_size):
            idx = order[start:start + minibatch_size]
            outputs = policy(
                gf[idx], ss[idx], sp[idx],
                ks[idx], kp[idx],
                ts[idx], tp[idx], tm[idx],
            )
            new_log_prob, entropy = action_log_prob_and_entropy(
                outputs, target_idx[idx], frac_bin[idx],
            )
            ratio = (new_log_prob - old_log_prob[idx]).exp()
            adv = advantages[idx]

            policy_loss = torch.maximum(
                -adv * ratio,
                -adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
            ).mean()
            # Clipped value loss — prevents large value updates
            value_pred = outputs.value
            value_clipped = old_values[idx] + torch.clamp(
                value_pred - old_values[idx], -clip_coef, clip_coef,
            )
            vl_unclipped = (returns[idx] - value_pred).pow(2)
            vl_clipped = (returns[idx] - value_clipped).pow(2)
            value_loss = 0.5 * torch.maximum(vl_unclipped, vl_clipped).mean()
            entropy_mean = entropy.mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean

            # Imitation loss blending
            im_loss_val = 0.0
            if use_imitation:
                from .imitation import compute_bc_loss
                demo_batch = demo_buffer.sample_batch(minibatch_size, device)
                im_loss = compute_bc_loss(policy, demo_batch)
                loss = loss + imitation_coef * im_loss
                im_loss_val = float(im_loss.detach().cpu())

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            metrics["loss"] += float(loss.detach().cpu())
            metrics["policy_loss"] += float(policy_loss.detach().cpu())
            metrics["value_loss"] += float(value_loss.detach().cpu())
            metrics["entropy"] += float(entropy_mean.detach().cpu())
            metrics["imitation_loss"] += im_loss_val
            updates += 1

    return {k: v / max(updates, 1) for k, v in metrics.items()}
