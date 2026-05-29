"""Per-planet lookahead search for V2 Expert Iteration.

For each owned source planet, evaluate candidate (target, ship-fraction) actions
by forward-simulating the *ground-truth* game model (src.simulator) and scoring
the resulting position. Returns improved distributions aligned with OrbitNet's
output layout:

  target_probs:  [P+1]      softmax over [hold, target_slot_0..P-1]
  frac_probs:    [P, K]     per target slot, softmax over ship-fraction bins

These serve as supervised distillation targets (the "expert" in ExIt is search).
"""
from __future__ import annotations

import numpy as np

from src.features import intercept_pos, passes_through_sun
from src.game_types import GameState
from src.simulator import (
    SimState,
    add_fleet_event,
    evaluate_state,
    sim_step,
    travel_time,
)

from .config import V2EnvConfig, V2ExItConfig
from .features import V2Features

NEG = -1e9


def _masked_softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    valid = scores > -1e8
    if not np.any(valid):
        out = np.zeros_like(scores)
        out[0] = 1.0
        return out
    shifted = np.where(valid, scores / max(temperature, 1e-6), NEG)
    shifted = shifted - shifted[valid].max()
    exp = np.where(valid, np.exp(shifted), 0.0)
    total = exp.sum()
    if total < 1e-10:
        out = np.zeros_like(scores)
        out[0] = 1.0
        return out
    return exp / total


def search_improve_planet(
    state: GameState,
    features: V2Features,
    sim_state: SimState,
    player: int,
    source_slot: int,
    env_cfg: V2EnvConfig,
    exit_cfg: V2ExItConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Improved (target_probs[P+1], frac_probs[P,K]) for one owned source planet."""
    P = env_cfg.max_planets
    fracs = env_cfg.ship_fractions
    K = len(fracs)

    target_scores = np.full(P + 1, NEG, dtype=np.float32)   # [hold, targets...]
    frac_scores = np.full((P, K), NEG, dtype=np.float32)

    # Hold baseline
    noop = sim_state.copy()
    for _ in range(exit_cfg.search_depth):
        sim_step(noop)
    target_scores[0] = evaluate_state(noop, player)

    src = features.planet_states[source_slot]
    if src is None or src.ships <= 0:
        return _make_dists(target_scores, frac_scores, exit_cfg.search_temperature, P, K)

    src_ships = src.ships
    candidates = [j for j in range(P) if features.reachability_mask[source_slot, j]]
    candidates = candidates[: exit_cfg.search_candidates]

    for j in candidates:
        tgt = features.planet_states[j]
        if tgt is None:
            continue
        if tgt.is_orbiting and state.angular_velocity > 0:
            tx, ty, _ = intercept_pos(src, tgt, src_ships, state.step, state.angular_velocity)
        else:
            tx, ty = tgt.x, tgt.y
        if passes_through_sun(src.x, src.y, tx, ty):
            continue

        best = NEG
        for fb, frac in enumerate(fracs):
            ships = max(1, int(src_ships * frac))
            tt = travel_time(src.x, src.y, tx, ty, ships)
            sc = sim_state.copy()
            add_fleet_event(sc, src.id, tgt.id, ships, tt)
            for _ in range(exit_cfg.search_depth):
                sim_step(sc)
            score = evaluate_state(sc, player)
            frac_scores[j, fb] = score
            if score > best:
                best = score
        target_scores[j + 1] = best

    return _make_dists(target_scores, frac_scores, exit_cfg.search_temperature, P, K)


def _make_dists(
    target_scores: np.ndarray,
    frac_scores: np.ndarray,
    temperature: float,
    P: int,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_probs = _masked_softmax(target_scores, temperature)
    frac_probs = np.zeros((P, K), dtype=np.float32)
    for j in range(P):
        row = frac_scores[j]
        if np.any(row > -1e8):
            frac_probs[j] = _masked_softmax(row, temperature)
        else:
            frac_probs[j] = 1.0 / K
    return target_probs, frac_probs
