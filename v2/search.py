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

import math

import numpy as np

from src.features import intercept_pos, passes_through_sun, planet_pos_at
from src.game_types import FleetState, GameState, PlanetState
from src.simulator import (
    SimState,
    add_fleet_event,
    evaluate_state,
    fleet_position_at,
    sim_step,
    travel_time,
)

from .config import V2EnvConfig, V2ExItConfig
from .features import V2Features, encode_features

NEG = -1e9


def _reconstruct_leaf_state(
    root_state: GameState,
    leaf_sim: SimState,
    player: int,
) -> GameState:
    """Reconstruct GameState at a search leaf for neural-value scoring.

    SimState is positionless, so planet geometry (x/y/radius/orbit) comes from the
    ROOT GameState; ownership/ships/production come from the leaf SimState.
    Orbiting planets are advanced to their leaf-step position.

    Phase 2 (positional simulator): in-flight fleets are now rebuilt with x/y/angle
    from the geometry each fleet_event carries (`fleet_position_at`), so the leaf
    GameState — and the features encoded from it — match the value head's training
    distribution. Dropping fleets (the old behaviour) made leaves OOD and collapsed
    the neural-value experiment 77%->0%; this is the prerequisite fix.
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
        planets.append(
            PlanetState(
                id=pid,
                owner=leaf_sim.planet_owner[pid],
                x=x,
                y=y,
                radius=rp.radius,
                ships=int(leaf_sim.planet_ships[pid]),
                production=leaf_sim.planet_prod[pid],
                is_orbiting=rp.is_orbiting,
                orbital_radius=rp.orbital_radius,
                initial_angle=rp.initial_angle,
            )
        )
    fleets: list[FleetState] = []
    for ev in leaf_sim.fleet_events:
        pos = fleet_position_at(ev, leaf_sim.current_step)
        if pos is None:
            continue
        fx, fy, fangle = pos
        fleets.append(
            FleetState(
                id=-1,
                owner=ev[2],
                x=fx,
                y=fy,
                angle=fangle,
                from_planet_id=-1,
                ships=int(ev[3]),
            )
        )
    return GameState(
        step=leaf_sim.current_step,
        player=player,
        planets=planets,
        fleets=fleets,
        angular_velocity=av,
        planets_by_id={p.id: p for p in planets},
    )


def _batch_neural_values(
    value_model,
    env_cfg: V2EnvConfig,
    leaf_states: list[GameState],
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


# ── Shared candidate enumeration + leaf simulation ───────────────────────────
#
# A candidate descriptor is one of:
#   ("hold", -1, -1, 0,     0.0, None,       None,       -1)
#   ("frac", j,  fb, ships,  tt,  (sx, sy),  (tx, ty),   tgt_id)
# Both the legacy (softmax-over-heuristic) and the Gumbel path enumerate the SAME
# candidates in the SAME order, so the two paths are directly comparable; the
# legacy path simulates them all eagerly, the Gumbel path lazily (only survivors).


def _enumerate_candidates(
    state: GameState,
    features: V2Features,
    source_slot: int,
    env_cfg: V2EnvConfig,
    exit_cfg: V2ExItConfig,
):
    """(src_planet, [descriptor, ...]) for one owned source planet. Hold first."""
    P = env_cfg.max_planets
    fracs = env_cfg.ship_fractions
    descs: list[tuple] = [("hold", -1, -1, 0, 0.0, None, None, -1)]
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
                descs.append(("frac", j, fb, ships, tt, (src.x, src.y), (tx, ty), tgt.id))
    return src, descs


def _decision_depth(descs: list[tuple], exit_cfg: V2ExItConfig) -> int:
    """Sim depth for ONE source-planet decision (Phase 2.2c arrival horizon).

    Flag off -> the fixed exit_cfg.search_depth (byte-identical legacy behaviour).
    Flag on -> min(cap, ceil(max tt over the enumerated candidates) + settle
    margin), so every candidate's fleet ARRIVES and the capture resolves before
    the leaf is scored. The depth is UNIFORM across the decision's candidates,
    including hold: evaluate_state's production term grows with sim depth, so
    leaves at different depths are not comparable — a deeper leaf would win/lose
    on accumulated production, not on the action.
    """
    if not bool(getattr(exit_cfg, "arrival_horizon", False)):
        return int(exit_cfg.search_depth)
    max_tt = max((d[4] for d in descs if d[0] == "frac"), default=0.0)
    margin = int(getattr(exit_cfg, "arrival_settle_margin", 4))
    cap = int(getattr(exit_cfg, "arrival_horizon_cap", 60))
    return min(cap, int(math.ceil(max_tt)) + margin)


def _advance(
    sc: SimState,
    depth: int,
    rollout_players: list[int] | None,
    launch_fn=None,
    every: int = 1,
) -> None:
    """Forward-sim `sc` `depth` steps. With no `launch_fn` this is exactly the
    original `for _ in range(depth): sim_step(sc, rollout_players)` (so the legacy
    path is byte-identical). With a `launch_fn` (Build 2 net opponent) the
    opponent acts only every `every`-th step (compute subsampling); on the
    skipped steps NO player launches, so we never silently fall back to the weak
    heuristic rollout."""
    if launch_fn is None:
        for _ in range(depth):
            sim_step(sc, rollout_players)
        return
    for d in range(depth):
        if d % every == 0:
            sim_step(sc, rollout_players, launch_fn)
        else:
            sim_step(sc, None, None)


def _simulate_descriptor(
    desc: tuple,
    sim_state: SimState,
    src_id: int,
    depth: int,
    rollout_players: list[int] | None,
    launch_fn,
    every: int,
) -> SimState:
    """Forward-sim one candidate descriptor from `sim_state`, returning the leaf."""
    sc = sim_state.copy()
    if desc[0] == "frac":
        _, _j, _fb, ships, tt, sxy, dxy, tid = desc
        add_fleet_event(sc, src_id, tid, ships, tt, src_xy=sxy, dst_xy=dxy)
    _advance(sc, depth, rollout_players, launch_fn, every)
    return sc


# ── Build 2: roll out the current distilled net as the in-sim opponent ────────


def _make_net_launch(root_state: GameState, env_cfg: V2EnvConfig, model):
    """Build a `launch_fn(sim, player) -> [(from_id, target_id, ships, tt), ...]`
    that drives `player` with the distilled net. The net runs on the in-distribution
    GameState reconstructed from the leaf SimState (`_reconstruct_leaf_state`, the
    same path the value head uses), so the opponent plays the real current policy
    rather than a hand-coded heuristic.

    Compute note: one full forward (logits + frac heads, not value_only) per
    opponent step per leaf — this is the expensive part Gumbel/Sequential-Halving
    exists to bound. `net_opponent_every` further subsamples it.
    """
    import torch

    from .actions import decode_actions
    from .state import predict_fleet_destination

    dev = next(model.parameters()).device

    def launch(sim: SimState, pl: int) -> list[tuple[int, int, int, float]]:
        gs = _reconstruct_leaf_state(root_state, sim, pl)
        feats = encode_features(gs, env_cfg)
        if not feats.own_mask.any():
            return []
        with torch.inference_mode():
            pf = torch.from_numpy(feats.planet_features).unsqueeze(0).to(dev)
            gf = torch.from_numpy(feats.global_features).unsqueeze(0).to(dev)
            pm = torch.from_numpy(feats.planet_mask).unsqueeze(0).to(dev)
            om = torch.from_numpy(feats.own_mask).unsqueeze(0).to(dev)
            rm = torch.from_numpy(feats.reachability_mask).unsqueeze(0).to(dev)
            out = model(pf, gf, pm, om, rm)
        moves = decode_actions(out, feats, gs, env_cfg, deterministic=True)
        by_id = {p.id: p for p in gs.planets}
        result: list[tuple[int, int, int, float]] = []
        for mv in moves:
            try:
                from_id, angle, ships = int(mv[0]), float(mv[1]), int(mv[2])
            except (TypeError, ValueError, IndexError):
                continue
            fp = by_id.get(from_id)
            if fp is None or ships <= 0:
                continue
            sx = fp.x + (fp.radius + 0.1) * math.cos(angle)
            sy = fp.y + (fp.radius + 0.1) * math.sin(angle)
            vf = FleetState(
                id=-1, owner=pl, x=sx, y=sy, angle=angle, from_planet_id=from_id, ships=ships
            )
            tgt, eta = predict_fleet_destination(vf, gs.planets, gs.step, gs.angular_velocity)
            if tgt is not None and math.isfinite(eta):
                result.append((from_id, int(tgt.id), ships, float(eta)))
        return result

    return launch


def _setup_opponent(
    state: GameState,
    sim_state: SimState,
    player: int,
    env_cfg: V2EnvConfig,
    exit_cfg: V2ExItConfig,
    value_model,
):
    """Resolve (rollout_players, launch_fn, every) for the in-sim opponent.

    Priority: net_opponent (Build 2, strong net) > rollout_search (Phase 1, weak
    heuristic, kept for reference) > none. Both flags off -> (None, None, 1) so
    the lookahead has NO opponent (the passive sim that produced the champion).
    """
    every = max(1, int(getattr(exit_cfg, "net_opponent_every", 1)))
    net_op = bool(getattr(exit_cfg, "net_opponent", False)) and value_model is not None
    if net_op and sim_state.planet_xy is not None:
        owners = {o for o in sim_state.planet_owner.values() if o >= 0}
        if not bool(getattr(exit_cfg, "net_opponent_self", True)):
            owners.discard(player)
        return sorted(owners), _make_net_launch(state, env_cfg, value_model), every
    if bool(getattr(exit_cfg, "rollout_search", False)) and sim_state.planet_xy is not None:
        owners = {o for o in sim_state.planet_owner.values() if o >= 0}
        if not bool(getattr(exit_cfg, "rollout_self", True)):
            owners.discard(player)
        return sorted(owners), None, every
    return None, None, every


# ── Build 1: Gumbel-Top-m + Sequential-Halving improved policy ────────────────


def _np_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _minmax(d: dict[int, float]) -> dict[int, float]:
    """Min-max normalize a {candidate: q} map to ~[0,1] (0.5 if degenerate)."""
    vals = np.array(list(d.values()), dtype=np.float64)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-12:
        return {c: 0.5 for c in d}
    return {c: (v - lo) / (hi - lo) for c, v in d.items()}


def _gumbel_search_planet(
    state: GameState,
    features: V2Features,
    sim_state: SimState,
    player: int,
    source_slot: int,
    env_cfg: V2EnvConfig,
    exit_cfg: V2ExItConfig,
    prior_target: np.ndarray,
    prior_frac: np.ndarray,
    rng_seed: int,
    rollout_players: list[int] | None,
    launch_fn,
    every: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gumbel AlphaZero (root-only) improved policy for one source planet.

    1. Prior logit of candidate a: log p_target(a) + log p_frac(a) (hold uses only
       the target prior). The prior is OrbitNet's collection-time forward pass.
    2. Draw g(a) ~ Gumbel(0,1); Gumbel-Top-m = take the m candidates with the
       largest logits(a)+g(a) (sampling WITHOUT replacement).
    3. Sequential Halving over the simulation budget: repeatedly forward-sim the
       survivors (heuristic evaluate_state leaf), keep the top half by
       logits(a)+g(a)+sigma(q_norm(a)), until one remains. Leaf rollouts are
       DETERMINISTIC, so each leaf is evaluated once and cached — the SH rounds
       re-rank cached q's and accumulate per-candidate "visits" for sigma.
    4. Distillation target = softmax(logits(a) + sigma(completedQ(a))) over ALL
       candidates, where completedQ = q_norm for simulated candidates and a
       prior-weighted v_mix completion for un-simulated ones (Danihelka 2022).
       This is a PROVABLE improvement over the policy prior — the anchor the raw
       softmax-over-heuristic lacked.

    OPEN QUESTION (do not block on it): Gumbel's improvement guarantee is per-node;
    we apply it decoupled per planet (DUCT-style). DUCT is empirically the strongest
    simultaneous-move MCTS variant, but the joint-turn guarantee is unproven. If
    results are flat, suspect the decoupling first.
    """
    P = env_cfg.max_planets
    K = len(env_cfg.ship_fractions)
    src, descs = _enumerate_candidates(state, features, source_slot, env_cfg, exit_cfg)
    src_id = src.id if src is not None else -1
    C = len(descs)

    # Prior logit per candidate (anchored selection).
    logits = np.empty(C, dtype=np.float64)
    for c, d in enumerate(descs):
        if d[0] == "hold":
            logits[c] = math.log(max(float(prior_target[0]), 1e-12))
        else:
            j, fb = d[1], d[2]
            logits[c] = math.log(max(float(prior_target[j + 1]), 1e-12)) + math.log(
                max(float(prior_frac[j, fb]), 1e-12)
            )

    rng = np.random.default_rng(rng_seed)
    g = rng.gumbel(0.0, 1.0, size=C)

    depth = _decision_depth(descs, exit_cfg)
    c_visit = float(exit_cfg.gumbel_c_visit)
    c_scale = float(exit_cfg.gumbel_c_scale)
    qcache: dict[int, float] = {}
    visits = np.zeros(C, dtype=np.float64)

    def sim_q(c: int) -> float:
        if c not in qcache:
            leaf = _simulate_descriptor(
                descs[c], sim_state, src_id, depth, rollout_players, launch_fn, every
            )
            qcache[c] = float(evaluate_state(leaf, player))
        return qcache[c]

    m = max(1, min(int(exit_cfg.gumbel_top_m), C))
    budget = max(int(exit_cfg.gumbel_sims), m)

    # Gumbel-Top-m (sampling without replacement = top-m by logits + g).
    order = np.argsort(-(logits + g))
    cur = [int(i) for i in order[:m]]

    if len(cur) > 1:
        num_phases = max(1, int(math.ceil(math.log2(len(cur)))))
        while len(cur) > 1:
            per = max(1, budget // (num_phases * len(cur)))
            for c in cur:
                sim_q(c)
                visits[c] += per
            qn = _minmax({c: qcache[c] for c in cur})
            maxv = max(float(visits.max()), 1.0)
            cur = sorted(
                cur,
                key=lambda c: logits[c] + g[c] + (c_visit + maxv) * c_scale * qn[c],
                reverse=True,
            )[: max(1, len(cur) // 2)]
    else:
        for c in cur:
            sim_q(c)
            visits[c] += 1

    # completedQ: simulated -> q_norm; un-simulated -> prior-weighted v_mix.
    sim_ids = list(qcache.keys())
    qn_all = _minmax({c: qcache[c] for c in sim_ids})
    pri = _np_softmax(logits)
    wsum = float(sum(pri[c] for c in sim_ids))
    vmix = float(sum(pri[c] * qn_all[c] for c in sim_ids) / wsum) if wsum > 1e-12 else 0.5
    completed = np.array([qn_all.get(c, vmix) for c in range(C)], dtype=np.float64)
    maxv = max(float(visits.max()), 1.0)
    pi = _np_softmax(logits + (c_visit + maxv) * c_scale * completed)

    # Factor the joint pi' back into target-marginal x per-target fraction.
    target_probs = np.zeros(P + 1, dtype=np.float32)
    frac_probs = np.full((P, K), 1.0 / K, dtype=np.float32)
    frac_acc: dict[int, np.ndarray] = {}
    for c, d in enumerate(descs):
        if d[0] == "hold":
            target_probs[0] += pi[c]
        else:
            j, fb = d[1], d[2]
            target_probs[j + 1] += pi[c]
            frac_acc.setdefault(j, np.zeros(K, dtype=np.float64))[fb] += pi[c]
    for j, arr in frac_acc.items():
        s = float(arr.sum())
        if s > 1e-12:
            frac_probs[j] = (arr / s).astype(np.float32)
    return target_probs, frac_probs


def search_improve_planet(
    state: GameState,
    features: V2Features,
    sim_state: SimState,
    player: int,
    source_slot: int,
    env_cfg: V2EnvConfig,
    exit_cfg: V2ExItConfig,
    value_model=None,
    prior_target: np.ndarray | None = None,
    prior_frac: np.ndarray | None = None,
    rng_seed: int = 0,
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
    blend_w = float(getattr(exit_cfg, "value_leaf_blend", 0.0))
    # Phase 3 blend is the middle path: only when a positive weight is set, a model
    # is supplied, and we are NOT in the pure-neural (swap) path.
    use_blend = (blend_w > 0.0) and (value_model is not None) and (not use_neural)

    # In-sim opponent (Build 2 net opponent > Phase 1 heuristic rollout > none).
    # When neither flag is set this is (None, None, *) and the lookahead is the
    # original passive sim — byte-identical to the champion.
    rollout_players, launch_fn, every = _setup_opponent(
        state, sim_state, player, env_cfg, exit_cfg, value_model
    )

    # Build 1: Gumbel/Sequential-Halving improved policy (heuristic leaf). Needs
    # the per-source policy prior threaded in from collection. Takes precedence
    # over the (refuted) value-blend; leaves stay heuristic per the research.
    if (
        bool(getattr(exit_cfg, "gumbel_search", False))
        and prior_target is not None
        and prior_frac is not None
    ):
        return _gumbel_search_planet(
            state,
            features,
            sim_state,
            player,
            source_slot,
            env_cfg,
            exit_cfg,
            prior_target,
            prior_frac,
            rng_seed,
            rollout_players,
            launch_fn,
            every,
        )

    target_scores = np.full(P + 1, NEG, dtype=np.float32)  # [hold, targets...]
    frac_scores = np.full((P, K), NEG, dtype=np.float32)

    # Collect forward-simulated leaves as (kind, j, fb, leaf_sim) over the shared
    # candidate enumeration (legacy path simulates them all eagerly).
    src, descs = _enumerate_candidates(state, features, source_slot, env_cfg, exit_cfg)
    src_id = src.id if src is not None else -1
    depth = _decision_depth(descs, exit_cfg)
    leaves: list[tuple[str, int, int, SimState]] = [
        (
            d[0],
            d[1],
            d[2],
            _simulate_descriptor(d, sim_state, src_id, depth, rollout_players, launch_fn, every),
        )
        for d in descs
    ]

    # Score all leaves: pure-neural (swap), blend, or heuristic per-leaf.
    if use_neural:
        leaf_states = [_reconstruct_leaf_state(state, lf, player) for (_, _, _, lf) in leaves]
        scores = _batch_neural_values(value_model, env_cfg, leaf_states)
    elif use_blend:
        # Score each leaf with BOTH the heuristic and OrbitNet's value head, then
        # z-score each across THIS decision's leaves and combine scale-free. The
        # z-scoring is what makes the two scorers commensurable (their raw ranges
        # differ); it is applied per source-planet decision, never globally.
        heur = np.asarray(
            [evaluate_state(lf, player) for (_, _, _, lf) in leaves],
            dtype=np.float64,
        )
        leaf_states = [_reconstruct_leaf_state(state, lf, player) for (_, _, _, lf) in leaves]
        neural = np.asarray(
            _batch_neural_values(value_model, env_cfg, leaf_states),
            dtype=np.float64,
        )
        zh = (heur - heur.mean()) / (heur.std() + 1e-6)
        zn = (neural - neural.mean()) / (neural.std() + 1e-6)
        scores = ((1.0 - blend_w) * zh + blend_w * zn).tolist()
    else:
        scores = [evaluate_state(lf, player) for (_, _, _, lf) in leaves]

    for (kind, j, fb, _), sc in zip(leaves, scores, strict=False):
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
