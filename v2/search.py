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

from src.features import intercept_pos, passes_through_sun, planet_pos_at
from src.game_types import GameState, PlanetState
from src.simulator import (
    SimState,
    add_fleet_event,
    evaluate_state,
    sim_step,
    travel_time,
)

from .config import V2EnvConfig, V2ExItConfig
from .features import V2Features, encode_features

NEG = -1e9


def _reconstruct_leaf_state(
    root_state: GameState, leaf_sim: SimState, player: int,
) -> GameState:
    """Approximate GameState at a search leaf for neural-value scoring.

    SimState is positionless, so geometry (x/y/radius/orbit) comes from the ROOT
    GameState; ownership/ships/production come from the leaf SimState. Orbiting
    planets are advanced to their leaf-step position. In-flight fleets are dropped
    (a leaf value estimate — the planet ownership/garrison is the dominant signal).
    """
    root_by_id = {p.id: p for p in root_state.planets}
    av = root_state.angular_velocity
    planets: list[PlanetState] = []
    for pid in leaf_sim.planet_ids:
        rp = root_by_id.get(pid)
        if rp is None:
            continue
        if rp.is_orbiting and av > 0:
            x, y = planet_pos_at(rp, leaf_sim.current_step, av)
        else:
            x, y = rp.x, rp.y
        planets.append(PlanetState(
            id=pid, owner=leaf_sim.planet_owner[pid], x=x, y=y, radius=rp.radius,
            ships=int(leaf_sim.planet_ships[pid]), production=leaf_sim.planet_prod[pid],
            is_orbiting=rp.is_orbiting, orbital_radius=rp.orbital_radius,
            initial_angle=rp.initial_angle,
        ))
    return GameState(
        step=leaf_sim.current_step, player=player, planets=planets, fleets=[],
        angular_velocity=av, planets_by_id={p.id: p for p in planets},
    )


def _batch_neural_values(
    value_model, env_cfg: V2EnvConfig, leaf_states: list[GameState],
) -> list[float]:
    """Score leaf GameStates with OrbitNet's value head in one batched pass.

    Uses `value_only` (trunk -> pool -> value), which skips the O(P^2) pairwise
    output heads — those dominate a full forward and are irrelevant to the value,
    so this is ~30-100x faster on CPU (the difference between a viable and an
    unviable neural-search ExIt run).
    """
    import numpy as _np
    import torch

    dev = next(value_model.parameters()).device
    feats = [encode_features(s, env_cfg) for s in leaf_states]
    pf = torch.from_numpy(_np.stack([f.planet_features for f in feats])).to(dev)
    gf = torch.from_numpy(_np.stack([f.global_features for f in feats])).to(dev)
    pm = torch.from_numpy(_np.stack([f.planet_mask for f in feats])).to(dev)
    with torch.inference_mode():
        vals = value_model.value_only(pf, gf, pm)
    return vals.detach().cpu().numpy().astype(float).tolist()


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
    value_model=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Improved (target_probs[P+1], frac_probs[P,K]) for one owned source planet.

    Leaves are scored by the handcrafted `evaluate_state`, or — when
    `exit_cfg.neural_value_leaves` is set and a `value_model` is supplied
    (Tier 3.2) — by OrbitNet's value head (a stronger expert). All candidate
    leaves are forward-simulated first, then scored in one batched pass.
    """
    P = env_cfg.max_planets
    fracs = env_cfg.ship_fractions
    K = len(fracs)
    use_neural = bool(getattr(exit_cfg, "neural_value_leaves", False)) and value_model is not None

    # Phase 1: every-step in-sim rollout opponent. All present players (optionally
    # incl. our own continuation) launch via the cheap rollout policy at each step,
    # so "hold" plays on and aggression is scored against a persistent opponent.
    rollout_players: list[int] | None = None
    if bool(getattr(exit_cfg, "rollout_search", False)) and sim_state.planet_xy is not None:
        owners = {o for o in sim_state.planet_owner.values() if o >= 0}
        if not bool(getattr(exit_cfg, "rollout_self", True)):
            owners.discard(player)
        rollout_players = sorted(owners)

    target_scores = np.full(P + 1, NEG, dtype=np.float32)   # [hold, targets...]
    frac_scores = np.full((P, K), NEG, dtype=np.float32)

    # Collect forward-simulated leaves as (kind, j, fb, leaf_sim).
    leaves: list[tuple[str, int, int, SimState]] = []

    # Hold baseline
    noop = sim_state.copy()
    for _ in range(exit_cfg.search_depth):
        sim_step(noop, rollout_players)
    leaves.append(("hold", -1, -1, noop))

    src = features.planet_states[source_slot]
    if src is not None and src.ships > 0:
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
            for fb, frac in enumerate(fracs):
                ships = max(1, int(src_ships * frac))
                tt = travel_time(src.x, src.y, tx, ty, ships)
                sc = sim_state.copy()
                add_fleet_event(sc, src.id, tgt.id, ships, tt)
                for _ in range(exit_cfg.search_depth):
                    sim_step(sc, rollout_players)
                leaves.append(("frac", j, fb, sc))

    # Score all leaves (neural batched, or heuristic per-leaf).
    if use_neural:
        leaf_states = [_reconstruct_leaf_state(state, lf, player) for (_, _, _, lf) in leaves]
        scores = _batch_neural_values(value_model, env_cfg, leaf_states)
    else:
        scores = [evaluate_state(lf, player) for (_, _, _, lf) in leaves]

    for (kind, j, fb, _), sc in zip(leaves, scores):
        if kind == "hold":
            target_scores[0] = sc
        else:
            frac_scores[j, fb] = sc
    # target score per planet = best fraction
    for j in range(P):
        if np.any(frac_scores[j] > -1e8):
            target_scores[j + 1] = frac_scores[j].max()

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
