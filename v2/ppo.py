"""PPO algorithm for V2 pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .actions import action_log_prob_and_entropy
from .model import OrbitNet


@dataclass
class V2TransitionBatch:
    planet_features: torch.Tensor    # [N, P, F]
    global_features: torch.Tensor    # [N, G]
    planet_mask: torch.Tensor        # [N, P] bool
    own_mask: torch.Tensor           # [N, P] bool
    target_indices: torch.Tensor     # [N, P] long
    log_prob: torch.Tensor           # [N]
    returns: torch.Tensor            # [N]
    advantages: torch.Tensor         # [N]
    values: torch.Tensor             # [N]


def v2_ppo_update(
    model: OrbitNet,
    optimizer: torch.optim.Optimizer,
    batch: V2TransitionBatch,
    *,
    clip_coef: float,
    ent_coef: float,
    vf_coef: float,
    max_grad_norm: float,
    epochs: int,
    minibatch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Clipped PPO update for V2 pipeline."""
    N = batch.planet_features.shape[0]
    if N < 4:
        return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

    # Move to device
    pf = batch.planet_features.to(device)
    gf = batch.global_features.to(device)
    pm = batch.planet_mask.to(device).bool()
    om = batch.own_mask.to(device).bool()
    ti = batch.target_indices.to(device)
    old_log_prob = batch.log_prob.to(device)
    returns = batch.returns.to(device)
    advantages = batch.advantages.to(device)
    old_values = batch.values.to(device)

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    minibatch_size = min(N, max(1, minibatch_size))
    metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    updates = 0

    for _ in range(epochs):
        order = torch.randperm(N, device=device)
        for start in range(0, N, minibatch_size):
            idx = order[start:start + minibatch_size]

            output = model(pf[idx], gf[idx], pm[idx], om[idx])

            new_log_prob, entropy = action_log_prob_and_entropy(
                output, om[idx], ti[idx],
            )

            # Importance ratio
            ratio = (new_log_prob - old_log_prob[idx]).exp()
            adv = advantages[idx]

            # Clipped policy loss
            policy_loss = torch.maximum(
                -adv * ratio,
                -adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef),
            ).mean()

            # Clipped value loss
            value_pred = output.value
            value_clipped = old_values[idx] + torch.clamp(
                value_pred - old_values[idx], -clip_coef, clip_coef,
            )
            vl_unclipped = (returns[idx] - value_pred).pow(2)
            vl_clipped = (returns[idx] - value_clipped).pow(2)
            value_loss = 0.5 * torch.maximum(vl_unclipped, vl_clipped).mean()

            entropy_mean = entropy.mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            metrics["loss"] += float(loss.detach().cpu())
            metrics["policy_loss"] += float(policy_loss.detach().cpu())
            metrics["value_loss"] += float(value_loss.detach().cpu())
            metrics["entropy"] += float(entropy_mean.detach().cpu())
            updates += 1

    return {k: v / max(updates, 1) for k, v in metrics.items()}
