"""Action sampling and decoding for V2 pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from src.features import BOARD_SIZE, fleet_speed, intercept_pos, passes_through_sun
from src.game_types import FleetState, GameState

from .config import V2EnvConfig
from .features import V2Features
from .model import OrbitNetOutput


@dataclass(slots=True)
class V2SampledAction:
    target_indices: torch.Tensor  # [B, P] sampled target per planet (1..P, 0 unused)
    log_prob: torch.Tensor        # [B] sum of per-planet log probs
    entropy: torch.Tensor         # [B] sum of per-planet entropies


def _mask_hold(logits: torch.Tensor) -> torch.Tensor:
    """Mask out the hold action (index 0) by setting it to -inf."""
    masked = logits.clone()
    masked[:, 0] = float("-inf")
    return masked


def sample_actions(
    output: OrbitNetOutput,
    own_mask: torch.Tensor,
    deterministic: bool = False,
) -> V2SampledAction:
    """Sample one target per owned planet from Categorical(logits).

    During training (deterministic=False), hold is masked out so every owned
    planet must send ships somewhere. This prevents exploration collapse where
    the model learns "never send ships" as a local optimum.

    Args:
        output: OrbitNetOutput with logits [B, P, P+1]
        own_mask: [B, P] bool (True = we own this planet)
        deterministic: If True, use argmax (eval mode, hold allowed).
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

        # Only process batch elements that own this planet to avoid NaN
        idx = slot_mask.nonzero(as_tuple=True)[0]
        logits_subset = output.logits[idx, i, :]  # [K, P+1]

        if deterministic:
            logits_safe = _safe_logits(logits_subset)
        else:
            # Mask hold (index 0) so model must pick a target
            logits_safe = _safe_logits(_mask_hold(logits_subset))

        dist = Categorical(logits=logits_safe)
        if deterministic:
            actions = logits_safe.argmax(dim=-1)  # [K]
        else:
            actions = dist.sample()  # [K]

        lp = dist.log_prob(actions)  # [K]
        ent = dist.entropy()  # [K]

        target_indices[idx, i] = actions
        log_probs[idx] += lp
        entropies[idx] += ent

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
    """Recompute log_prob and entropy for stored actions (PPO update).

    Must use the same hold-masking as sample_actions (training mode).
    Only processes batch elements that own planet i to avoid NaN from
    all-masked logits.
    """
    B, P, _ = output.logits.shape
    device = output.logits.device

    log_probs = torch.zeros(B, device=device)
    entropies = torch.zeros(B, device=device)

    for i in range(P):
        slot_mask = own_mask[:, i]  # [B]
        if not slot_mask.any():
            continue

        # Only process batch elements that own this planet
        idx = slot_mask.nonzero(as_tuple=True)[0]
        logits_subset = output.logits[idx, i, :]  # [K, P+1]
        logits_masked = _safe_logits(_mask_hold(logits_subset))
        dist = Categorical(logits=logits_masked)
        lp = dist.log_prob(target_indices[idx, i])
        ent = dist.entropy()

        log_probs[idx] += lp
        entropies[idx] += ent

    return log_probs, entropies


def decode_sampled_actions(
    sampled: V2SampledAction,
    output: OrbitNetOutput,
    features: V2Features,
    state: GameState,
    cfg: V2EnvConfig,
) -> list[list[float | int]]:
    """Convert already-sampled actions to Kaggle moves (training rollout).

    Uses the same target_indices stored in the PPO buffer, ensuring
    the executed actions match the recorded log_probs.
    """
    P = cfg.max_planets
    logits = output.logits[0]  # [P, P+1]
    moves: list[list[float | int]] = []

    for i in range(P):
        if not features.own_mask[i]:
            continue
        target_idx = int(sampled.target_indices[0, i].item())
        if target_idx == 0:  # hold (shouldn't happen with masking, but safe)
            continue

        j = target_idx - 1  # target planet slot
        src = features.planet_states[i]
        tgt = features.planet_states[j]
        if src is None or tgt is None or src.id == tgt.id or src.ships <= 0:
            continue

        # Ship fraction: softmax prob (hold-masked), clamped to [0.2, 1.0]
        row_logits = logits[i].clone()
        row_logits[0] = float("-inf")
        if not torch.isfinite(row_logits).any():
            continue
        probs = torch.softmax(row_logits, dim=-1)
        frac = max(0.2, min(1.0, float(probs[target_idx])))
        ships = int(math.floor(src.ships * frac))
        if ships < cfg.min_ships_to_send:
            continue

        # For non-owned targets: fleet must be large enough to capture
        if tgt.owner != state.player and ships <= tgt.ships:
            continue

        angle = _compute_angle(src, tgt, ships, state, cfg)
        if angle is not None:
            moves.append([src.id, angle, ships])

    return moves


def decode_actions(
    output: OrbitNetOutput,
    features: V2Features,
    state: GameState,
    cfg: V2EnvConfig,
    deterministic: bool = True,
) -> list[list[float | int]]:
    """Convert model output to Kaggle [planet_id, angle, ships] commands.

    Deterministic (eval): use full softmax (hold included). Send to targets
    above threshold. If no target exceeds threshold, send to the best target
    anyway (argmax fallback) so the agent always acts early in training.

    Stochastic (train): hold is masked out. Sample one target per planet,
    send a meaningful fraction of ships (conditional probability, min 20%).
    Note: prefer decode_sampled_actions() in the training loop so executed
    actions match the PPO buffer's log_probs.
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

        if deterministic:
            probs = torch.softmax(row_logits, dim=-1)  # [P+1]
            available_ships = src.ships
            made_move = False

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

                # For non-owned targets: fleet must be large enough to capture
                if tgt.owner != state.player and ships <= tgt.ships:
                    continue

                angle = _compute_angle(src, tgt, ships, state, cfg)
                if angle is not None:
                    moves.append([src.id, angle, ships])
                    made_move = True

            # Argmax fallback: if no target exceeded threshold, send to the
            # best non-hold target so the agent always acts early in training.
            if not made_move:
                target_probs = probs[1:].clone()  # [P]
                for j in range(P):
                    tgt = features.planet_states[j]
                    if tgt is None or (tgt.id == src.id):
                        target_probs[j] = 0.0
                if target_probs.sum() > 0:
                    best_j = int(target_probs.argmax().item())
                    tgt = features.planet_states[best_j]
                    if tgt is not None:
                        frac = max(0.2, float(target_probs[best_j]) / float(target_probs.sum().clamp(min=1e-8)))
                        ships = int(math.floor(available_ships * frac))
                        if ships >= cfg.min_ships_to_send:
                            # For non-owned targets: fleet must capture
                            if tgt.owner != state.player and ships <= tgt.ships:
                                pass  # skip undersized fleet
                            else:
                                angle = _compute_angle(src, tgt, ships, state, cfg)
                                if angle is not None:
                                    moves.append([src.id, angle, ships])
        else:
            # Mask hold, sample a target, send a meaningful ship fraction
            train_logits = row_logits.clone()
            train_logits[0] = float("-inf")

            if not torch.isfinite(train_logits).any():
                continue

            probs = torch.softmax(train_logits, dim=-1)  # [P+1], hold=0
            dist = Categorical(logits=train_logits)
            action = dist.sample().item()

            j = action - 1
            tgt = features.planet_states[j]
            if tgt is None or tgt.id == src.id:
                continue

            # Ship fraction: use probability among targets, min 20%
            frac = max(0.2, min(1.0, float(probs[action])))
            ships = int(math.floor(src.ships * frac))
            if ships < cfg.min_ships_to_send:
                continue

            # For non-owned targets: fleet must be large enough to capture
            if tgt.owner != state.player and ships <= tgt.ships:
                continue

            angle = _compute_angle(src, tgt, ships, state, cfg)
            if angle is not None:
                moves.append([src.id, angle, ships])

    return moves


def _compute_angle(
    src, tgt, ships: int, state: GameState, cfg: V2EnvConfig,
) -> float | None:
    """Compute launch angle ensuring fleet reaches the target.

    For orbiting targets, aims at the predicted intercept position.
    Validates the full flight path: sun intersection, map boundary,
    and that the fleet actually hits the intended target.
    """
    # For orbiting targets, use intercept position
    if tgt.is_orbiting and state.angular_velocity > 0:
        ix, iy, _ = intercept_pos(src, tgt, ships, state.step, state.angular_velocity)
    else:
        ix, iy = tgt.x, tgt.y

    # Direct angle to aim position (no waypoint redirect)
    angle = math.atan2(iy - src.y, ix - src.x)

    # Quick filter: direct path crosses the sun
    if passes_through_sun(src.x, src.y, ix, iy):
        return None

    # Full path validation: verify fleet actually hits the intended target
    if not _validate_will_hit(src, tgt, angle, ships, state):
        return None

    return float(angle)


def _validate_will_hit(
    src, tgt, angle: float, ships: int, state: GameState,
) -> bool:
    """Verify fleet launched at `angle` will actually hit the intended target.

    Creates a virtual fleet and simulates the path, checking for sun
    intersection, map boundary exit, and collision with other planets.
    """
    from .state import predict_fleet_destination

    # Fleet spawns just outside the source planet's surface
    spawn_x = src.x + (src.radius + 0.1) * math.cos(angle)
    spawn_y = src.y + (src.radius + 0.1) * math.sin(angle)

    # Spawn point must be in bounds
    if spawn_x <= 0 or spawn_x >= BOARD_SIZE or spawn_y <= 0 or spawn_y >= BOARD_SIZE:
        return False

    virtual_fleet = FleetState(
        id=-1, owner=state.player,
        x=spawn_x, y=spawn_y,
        angle=angle,
        from_planet_id=src.id,
        ships=ships,
    )

    # Exclude source planet from collision check (fleet just launched from it)
    other_planets = [p for p in state.planets if p.id != src.id]
    hit_planet, _eta = predict_fleet_destination(
        virtual_fleet, other_planets, state.step, state.angular_velocity,
    )

    return hit_planet is not None and hit_planet.id == tgt.id


def _safe_logits(logits: torch.Tensor) -> torch.Tensor:
    """Ensure at least one valid logit per row."""
    invalid_rows = ~torch.isfinite(logits).any(dim=-1)
    if not invalid_rows.any():
        return logits
    safe = logits.clone()
    # Fallback: enable first target (index 1) if all are -inf
    safe[invalid_rows, 1] = 0.0
    return safe
