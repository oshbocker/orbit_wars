"""Per-planet lookahead search for Expert Iteration.

For each planet decision, evaluates candidate actions by forward-simulating
the game state and scoring the resulting position. Returns improved target
distributions (softmax over scores) that serve as supervised training targets.
"""
from __future__ import annotations

import math

import numpy as np

from .config import ExItConfig, EnvConfig
from .features import (
    SourceDecision,
    fleet_speed,
    passes_through_sun,
    safe_angle,
    intercept_pos,
    BOARD_SIZE,
)
from .game_types import GameState, PlanetState, SUN_X, SUN_Y
from .simulator import SimState, add_fleet_event, evaluate_state, sim_step, travel_time


def _map_search_fracs_to_policy_bins(
    search_fractions: list[float],
    policy_fractions: list[float],
) -> list[int]:
    """Map each search fraction to the nearest policy fraction bin index."""
    mapping = []
    for sf in search_fractions:
        best_idx = 0
        best_dist = abs(sf - policy_fractions[0])
        for i, pf in enumerate(policy_fractions):
            d = abs(sf - pf)
            if d < best_dist:
                best_dist = d
                best_idx = i
        mapping.append(best_idx)
    return mapping


def search_improve_with_player(
    decision: SourceDecision,
    sim_state: SimState,
    game_state: GameState,
    player: int,
    step: int,
    exit_cfg: ExItConfig,
    env_cfg: EnvConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate candidate actions and return improved distributions.

    Returns:
        target_probs: [1+T] softmax distribution over NoOp + targets
        fraction_probs: [T, num_policy_fracs] softmax over policy fraction bins per target
    """
    n_targets = len(decision.target_planet_ids)
    T = len(decision.target_mask) - 2
    num_search_fracs = len(exit_cfg.search_fractions)
    num_policy_fracs = len(env_cfg.ship_fractions)

    # Map search fractions to policy fraction bin indices
    frac_mapping = _map_search_fracs_to_policy_bins(
        exit_cfg.search_fractions, env_cfg.ship_fractions,
    )

    scores = np.full(1 + T, -1e9, dtype=np.float32)
    # Scores in policy fraction space
    fraction_scores = np.full((T, num_policy_fracs), -1e9, dtype=np.float32)

    # NoOp
    noop_sim = sim_state.copy()
    for _ in range(exit_cfg.search_depth):
        sim_step(noop_sim)
    scores[0] = evaluate_state(noop_sim, player)

    src_ships = decision.remaining_ships
    if src_ships <= 0:
        return _make_distributions(scores, fraction_scores, exit_cfg.search_temperature, T, num_policy_fracs)

    src_id = decision.source_id
    planets_by_id = game_state.planets_by_id
    src_planet = planets_by_id.get(src_id)
    if src_planet is None:
        return _make_distributions(scores, fraction_scores, exit_cfg.search_temperature, T, num_policy_fracs)

    for tgt_idx in range(min(n_targets, exit_cfg.search_candidates)):
        if not decision.target_mask[tgt_idx + 2]:
            continue

        target_id = decision.target_planet_ids[tgt_idx]
        tgt_planet = planets_by_id.get(target_id)
        if tgt_planet is None:
            continue

        if tgt_planet.is_orbiting and game_state.angular_velocity > 0:
            tgt_x, tgt_y, _ = intercept_pos(
                src_planet, tgt_planet, src_ships,
                step, game_state.angular_velocity,
            )
        else:
            tgt_x, tgt_y = tgt_planet.x, tgt_planet.y

        if passes_through_sun(src_planet.x, src_planet.y, tgt_x, tgt_y):
            continue

        best_frac_score = -1e9
        for sf_idx, frac in enumerate(exit_cfg.search_fractions):
            ships_to_send = max(1, int(src_ships * frac))
            actual_tt = travel_time(src_planet.x, src_planet.y, tgt_x, tgt_y, ships_to_send)

            sim_copy = sim_state.copy()
            add_fleet_event(sim_copy, src_id, target_id, ships_to_send, actual_tt,
                            src_xy=(src_planet.x, src_planet.y), dst_xy=(tgt_x, tgt_y))

            for _ in range(exit_cfg.search_depth):
                sim_step(sim_copy)
            score = evaluate_state(sim_copy, player)

            # Map to policy fraction bin
            policy_bin = frac_mapping[sf_idx]
            # Keep best score if multiple search fractions map to same bin
            if score > fraction_scores[tgt_idx, policy_bin]:
                fraction_scores[tgt_idx, policy_bin] = score

            if score > best_frac_score:
                best_frac_score = score

        scores[tgt_idx + 1] = best_frac_score

    return _make_distributions(scores, fraction_scores, exit_cfg.search_temperature, T, num_policy_fracs)


def _make_distributions(
    scores: np.ndarray,
    fraction_scores: np.ndarray,
    temperature: float,
    T: int,
    num_fracs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw scores to probability distributions via softmax."""
    # Target distribution
    target_probs = _masked_softmax(scores, temperature)

    # Fraction distributions (per target)
    frac_probs = np.zeros((T, num_fracs), dtype=np.float32)
    for t in range(T):
        row = fraction_scores[t]
        if np.any(row > -1e8):
            frac_probs[t] = _masked_softmax(row, temperature)
        else:
            # Uniform over fractions if no valid scores
            frac_probs[t] = 1.0 / num_fracs

    return target_probs, frac_probs


def _masked_softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Softmax over valid (non -1e9) entries."""
    valid = scores > -1e8
    if not np.any(valid):
        out = np.zeros_like(scores)
        out[0] = 1.0  # default to first entry
        return out
    shifted = np.where(valid, scores / max(temperature, 1e-6), -1e9)
    shifted = shifted - shifted[valid].max()  # numerical stability
    exp = np.where(valid, np.exp(shifted), 0.0)
    total = exp.sum()
    if total < 1e-10:
        out = np.zeros_like(scores)
        out[0] = 1.0
        return out
    return exp / total
