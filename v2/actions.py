"""Action sampling and decoding for V2 pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from src.features import fleet_speed, intercept_pos, passes_through_sun, safe_angle
from src.game_types import GameState

from .config import V2EnvConfig
from .features import V2Features
from .model import OrbitNetOutput


@dataclass(slots=True)
class V2SampledAction:
    target_indices: torch.Tensor  # [B, P] sampled target per planet (0=hold)
    log_prob: torch.Tensor        # [B] sum of per-planet log probs
    entropy: torch.Tensor         # [B] sum of per-planet entropies


def sample_actions(
    output: OrbitNetOutput,
    own_mask: torch.Tensor,
    deterministic: bool = False,
) -> V2SampledAction:
    """Sample one target per owned planet from Categorical(logits).

    Args:
        output: OrbitNetOutput with logits [B, P, P+1]
        own_mask: [B, P] bool (True = we own this planet)
        deterministic: If True, use argmax instead of sampling.
    """
    B, P, _ = output.logits.shape
    device = output.logits.device

    target_indices = torch.zeros(B, P, dtype=torch.long, device=device)
    log_probs = torch.zeros(B, device=device)
    entropies = torch.zeros(B, device=device)

    for i in range(P):
        slot_mask = own_mask[:, i]  # [B] which batches own planet i
        if not slot_mask.any():
            continue

        logits_i = output.logits[:, i, :]  # [B, P+1]

        # Ensure at least one valid logit per row
        logits_safe = _safe_logits(logits_i)

        dist = Categorical(logits=logits_safe)
        if deterministic:
            actions = logits_safe.argmax(dim=-1)  # [B]
        else:
            actions = dist.sample()  # [B]

        lp = dist.log_prob(actions)  # [B]
        ent = dist.entropy()  # [B]

        # Only accumulate for planets we own
        target_indices[:, i] = actions * slot_mask.long()
        log_probs += lp * slot_mask.float()
        entropies += ent * slot_mask.float()

    return V2SampledAction(
        target_indices=target_indices,
        log_prob=log_probs,
        entropy=entropies,
    )


def action_log_prob_and_entropy(
    output: OrbitNetOutput,
    own_mask: torch.Tensor,
    target_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute log_prob and entropy for stored actions (PPO update)."""
    B, P, _ = output.logits.shape
    device = output.logits.device

    log_probs = torch.zeros(B, device=device)
    entropies = torch.zeros(B, device=device)

    for i in range(P):
        slot_mask = own_mask[:, i]
        if not slot_mask.any():
            continue

        logits_i = _safe_logits(output.logits[:, i, :])
        dist = Categorical(logits=logits_i)
        lp = dist.log_prob(target_indices[:, i])
        ent = dist.entropy()

        log_probs += lp * slot_mask.float()
        entropies += ent * slot_mask.float()

    return log_probs, entropies


def decode_actions(
    output: OrbitNetOutput,
    features: V2Features,
    state: GameState,
    cfg: V2EnvConfig,
    deterministic: bool = True,
) -> list[list[float | int]]:
    """Convert model output to Kaggle [planet_id, angle, ships] commands.

    Deterministic (eval): use full softmax. For each target with P > threshold,
    send floor(ships * P) ships. Sun avoidance + intercept for orbiting targets.

    This operates on a single observation (no batch dim).
    """
    P = cfg.max_planets
    logits = output.logits[0]  # [P, P+1]
    moves: list[list[float | int]] = []

    for i in range(P):
        if not features.own_mask[i]:
            continue
        src = features.planet_states[i]
        if src is None or src.ships <= 0:
            continue

        # Softmax over valid targets
        row_logits = logits[i]  # [P+1]
        # Check if any valid target exists (non -inf)
        finite_mask = torch.isfinite(row_logits)
        if not finite_mask.any():
            continue

        probs = torch.softmax(row_logits, dim=-1)  # [P+1]
        available_ships = src.ships

        if deterministic:
            # Full allocation: send to all targets above threshold
            for j in range(P):
                prob_j = float(probs[j + 1])  # +1 because index 0 = hold
                if prob_j < cfg.allocation_threshold:
                    continue
                tgt = features.planet_states[j]
                if tgt is None or tgt.id == src.id:
                    continue

                ships = int(math.floor(available_ships * prob_j))
                if ships < cfg.min_ships_to_send:
                    continue

                angle = _compute_angle(src, tgt, ships, state, cfg)
                if angle is not None:
                    moves.append([src.id, angle, ships])
        else:
            # Stochastic: sample one target, send ships
            # Use conditional probability (prob among non-hold options) as
            # the ship fraction. This avoids the cold-start problem where
            # uniform softmax over 41 options gives ~2.4% per target,
            # resulting in floor() = 0 ships for most planets.
            dist = Categorical(logits=row_logits)
            action = dist.sample().item()
            if action == 0:  # hold
                continue

            j = action - 1
            tgt = features.planet_states[j]
            if tgt is None or tgt.id == src.id:
                continue

            prob_hold = float(probs[0])
            prob_j = float(probs[action])
            # Conditional prob: P(target | not hold), clamped to [0.2, 1.0]
            cond_prob = prob_j / max(1.0 - prob_hold, 1e-6)
            frac = max(0.2, min(1.0, cond_prob))
            ships = int(math.floor(available_ships * frac))
            if ships < cfg.min_ships_to_send:
                continue

            angle = _compute_angle(src, tgt, ships, state, cfg)
            if angle is not None:
                moves.append([src.id, angle, ships])

    return moves


def _compute_angle(
    src, tgt, ships: int, state: GameState, cfg: V2EnvConfig,
) -> float | None:
    """Compute launch angle with sun avoidance and orbit intercept."""
    # For orbiting targets, use intercept position
    if tgt.is_orbiting and state.angular_velocity > 0:
        ix, iy, _ = intercept_pos(src, tgt, ships, state.step, state.angular_velocity)
    else:
        ix, iy = tgt.x, tgt.y

    # Sun avoidance
    angle, _ = safe_angle(src.x, src.y, ix, iy)

    # Verify the path doesn't go through the sun after redirection
    if passes_through_sun(src.x, src.y, ix, iy):
        return None

    return float(angle)


def _safe_logits(logits: torch.Tensor) -> torch.Tensor:
    """Ensure at least one valid logit per row."""
    invalid_rows = ~torch.isfinite(logits).any(dim=-1)
    if not invalid_rows.any():
        return logits
    safe = logits.clone()
    safe[invalid_rows, 0] = 0.0  # fallback to hold
    return safe
