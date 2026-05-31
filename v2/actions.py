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
    target_indices: torch.Tensor  # [B, P] sampled target per planet (1..P, 0=hold)
    frac_indices: torch.Tensor    # [B, P] sampled ship-fraction bin per planet (0..K-1)
    log_prob: torch.Tensor        # [B] sum of per-planet (target + fraction) log probs
    entropy: torch.Tensor         # [B] MEAN per-planet (target + fraction) entropy


def _mask_hold(logits: torch.Tensor) -> torch.Tensor:
    """Mask out the hold action (index 0) by setting it to -inf."""
    masked = logits.clone()
    masked[:, 0] = float("-inf")
    return masked


def _frac_dist_for(output: OrbitNetOutput, idx: torch.Tensor, i: int,
                   actions: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
    """Categorical over fraction bins for the chosen target of source planet i.

    Returns (distribution, send_mask) where send_mask is True for non-hold
    actions. For hold actions the target slot is clamped to 0 (the row is
    ignored downstream via send_mask).
    """
    tgt_slot = (actions - 1).clamp(min=0)                       # [M]
    fl = output.frac_logits[idx, i]                             # [M, P, K]
    rows = torch.arange(fl.shape[0], device=fl.device)
    frac_row = fl[rows, tgt_slot]                               # [M, K]
    return Categorical(logits=frac_row), (actions > 0)


def sample_actions(
    output: OrbitNetOutput,
    own_mask: torch.Tensor,
    deterministic: bool = False,
) -> V2SampledAction:
    """Sample a factored (target, ship-fraction) action per owned planet.

    During training (deterministic=False), hold is masked out so every owned
    planet must send ships somewhere. The ship fraction is a *separate*
    categorical over discrete bins (decoupled from target selection), so the
    policy gradient can learn fleet size directly.
    """
    B, P, _ = output.logits.shape
    device = output.logits.device

    target_indices = torch.zeros(B, P, dtype=torch.long, device=device)
    frac_indices = torch.zeros(B, P, dtype=torch.long, device=device)
    log_probs = torch.zeros(B, device=device)
    entropies = torch.zeros(B, device=device)

    for i in range(P):
        slot_mask = own_mask[:, i]
        if not slot_mask.any():
            continue
        idx = slot_mask.nonzero(as_tuple=True)[0]
        logits_subset = output.logits[idx, i, :]  # [M, P+1]

        if deterministic:
            logits_safe = _safe_logits(logits_subset)
            tdist = Categorical(logits=logits_safe)
            actions = logits_safe.argmax(dim=-1)
        else:
            logits_safe = _safe_logits(_mask_hold(logits_subset))
            tdist = Categorical(logits=logits_safe)
            actions = tdist.sample()

        t_lp = tdist.log_prob(actions)
        t_ent = tdist.entropy()

        # Factored ship-fraction
        fdist, send = _frac_dist_for(output, idx, i, actions)
        fbins = fdist.probs.argmax(dim=-1) if deterministic else fdist.sample()
        f_lp = fdist.log_prob(fbins) * send.float()
        f_ent = fdist.entropy() * send.float()
        fbins = fbins * send.long()

        target_indices[idx, i] = actions
        frac_indices[idx, i] = fbins
        log_probs[idx] += t_lp + f_lp
        entropies[idx] += t_ent + f_ent

    # Fix 2: mean per-planet entropy (decouple exploration pressure from planet count)
    own_counts = own_mask.sum(dim=1).clamp(min=1).float()
    entropies = entropies / own_counts

    return V2SampledAction(
        target_indices=target_indices,
        frac_indices=frac_indices,
        log_prob=log_probs,
        entropy=entropies,
    )


def action_log_prob_and_entropy(
    output: OrbitNetOutput,
    own_mask: torch.Tensor,
    target_indices: torch.Tensor,
    frac_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute factored (target + fraction) log_prob and mean entropy (PPO update).

    Mirrors sample_actions' training-mode masking. log_prob is summed over
    planets (joint action); entropy is the mean per owned planet.
    """
    B, P, _ = output.logits.shape
    device = output.logits.device

    log_probs = torch.zeros(B, device=device)
    entropies = torch.zeros(B, device=device)

    for i in range(P):
        slot_mask = own_mask[:, i]
        if not slot_mask.any():
            continue
        idx = slot_mask.nonzero(as_tuple=True)[0]
        logits_subset = output.logits[idx, i, :]
        logits_masked = _safe_logits(_mask_hold(logits_subset))
        tdist = Categorical(logits=logits_masked)
        actions = target_indices[idx, i]
        t_lp = tdist.log_prob(actions)
        t_ent = tdist.entropy()

        fdist, send = _frac_dist_for(output, idx, i, actions)
        f_lp = fdist.log_prob(frac_indices[idx, i]) * send.float()
        f_ent = fdist.entropy() * send.float()

        log_probs[idx] += t_lp + f_lp
        entropies[idx] += t_ent + f_ent

    own_counts = own_mask.sum(dim=1).clamp(min=1).float()
    entropies = entropies / own_counts
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

        # Ship fraction: dedicated fraction head (decoupled from target selection)
        frac_bin = int(sampled.frac_indices[0, i].item())
        frac_bin = max(0, min(len(cfg.ship_fractions) - 1, frac_bin))
        frac = cfg.ship_fractions[frac_bin]
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
    force_act: bool = False,
) -> list[list[float | int]]:
    """Convert model output to Kaggle [planet_id, angle, ships] commands.

    Factored action: each owned source picks ONE target (argmax in eval, sample
    in train) and ONE ship-fraction bin from the dedicated fraction head. This
    decouples fleet size from target-selection probability.

    force_act: in deterministic (eval) mode, mask the hold action so every owned
    planet sends (argmax over targets only). Diagnostic for the train/eval hold
    mismatch — PPO trains with hold masked, so the hold logit is BC-driven and
    may be spuriously inflated.
    """
    P = cfg.max_planets
    logits = output.logits[0]            # [P, P+1]
    frac_logits = output.frac_logits[0]  # [P, P, K]
    fracs = cfg.ship_fractions
    moves: list[list[float | int]] = []

    for i in range(P):
        if not features.own_mask[i]:
            continue
        src = features.planet_states[i]
        if src is None or src.ships <= 0:
            continue

        row_logits = logits[i]  # [P+1]
        if not torch.isfinite(row_logits).any():
            continue

        if deterministic:
            if force_act:
                # Mask hold: argmax over targets only (must send if any valid).
                masked = row_logits.clone()
                masked[0] = float("-inf")
                if not torch.isfinite(masked).any():
                    continue
                action = int(masked.argmax().item())
            else:
                # Hold allowed in eval: argmax over [hold, targets]
                action = int(row_logits.argmax().item())
                if action == 0:
                    continue  # model chose to hold
        else:
            train_logits = row_logits.clone()
            train_logits[0] = float("-inf")
            if not torch.isfinite(train_logits).any():
                continue
            action = int(Categorical(logits=train_logits).sample().item())

        j = action - 1
        tgt = features.planet_states[j]
        if tgt is None or tgt.id == src.id:
            continue

        # Ship fraction from the dedicated head (argmax in eval, sample in train)
        frac_row = frac_logits[i, j]  # [K]
        if deterministic:
            fbin = int(frac_row.argmax().item())
        else:
            fbin = int(Categorical(logits=frac_row).sample().item())
        fbin = max(0, min(len(fracs) - 1, fbin))
        ships = int(math.floor(src.ships * fracs[fbin]))
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
