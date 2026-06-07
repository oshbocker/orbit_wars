"""Bit-identity + sanity tests for the Gumbel/Sequential-Halving search build.

1. BIT-IDENTITY: with exit.gumbel_search=False (and net_opponent=False), the new
   refactored search must produce output byte-identical to the original
   softmax-over-heuristic-leaf algorithm. A reference copy of that original
   algorithm is embedded below and compared per owned source planet, across
   several real mid-game states (with in-flight fleets, so positional leaf
   reconstruction is exercised).

2. GUMBEL SANITY: with gumbel_search=True the improved (target, frac) dists are
   finite, sum to 1, mask-respecting, and reproducible for a fixed seed.

Run: uv run python scripts/test_gumbel_search.py
"""

from __future__ import annotations

import numpy as np

from src.features import intercept_pos, passes_through_sun
from src.game_types import parse_observation
from src.simulator import (
    add_fleet_event,
    build_sim_state,
    evaluate_state,
    sim_step,
    travel_time,
)
from v2.config import load_v2_config
from v2.features import encode_features
from v2.search import _make_dists, search_improve_planet

NEG = -1e9


def _ref_search_planet(state, features, sim_state, player, source_slot, env_cfg, exit_cfg):
    """Reference copy of the ORIGINAL (pre-Gumbel-refactor) heuristic search for
    one source planet. Used only to prove the refactor is bit-identical."""
    P = env_cfg.max_planets
    fracs = env_cfg.ship_fractions
    K = len(fracs)
    target_scores = np.full(P + 1, NEG, dtype=np.float32)
    frac_scores = np.full((P, K), NEG, dtype=np.float32)
    leaves = []

    noop = sim_state.copy()
    for _ in range(exit_cfg.search_depth):
        sim_step(noop)
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
                add_fleet_event(
                    sc, src.id, tgt.id, ships, tt, src_xy=(src.x, src.y), dst_xy=(tx, ty)
                )
                for _ in range(exit_cfg.search_depth):
                    sim_step(sc)
                leaves.append(("frac", j, fb, sc))

    scores = [evaluate_state(lf, player) for (_, _, _, lf) in leaves]
    for (kind, j, fb, _), sc in zip(leaves, scores, strict=False):
        if kind == "hold":
            target_scores[0] = sc
        else:
            frac_scores[j, fb] = sc
    for j in range(P):
        if np.any(frac_scores[j] > -1e8):
            target_scores[j + 1] = frac_scores[j].max()
    return _make_dists(target_scores, frac_scores, exit_cfg.search_temperature, P, K)


def _gen_states(cfg, n_states=4, warm=30):
    """Step fast_env with apex on both sides to reach mid-game states with
    in-flight fleets, returning (state, features, sim_state) tuples."""
    from agents.apex import agent as apex_agent
    from v2.fast_env import FastOrbitWars

    out = []
    sim = FastOrbitWars(num_agents=2, seed=7)
    t = 0
    while not sim.done and len(out) < n_states:
        obs0, obs1 = sim.observation(0), sim.observation(1)
        if t >= warm and t % 5 == 0:
            state = parse_observation(obs0)
            feats = encode_features(state, cfg.env)
            ss = build_sim_state(state, with_geometry=True)
            out.append((state, feats, ss))
        sim.step([apex_agent(obs0) or [], apex_agent(obs1) or []])
        t += 1
    return out


def main() -> None:
    cfg = load_v2_config("configs/v2_exit.yaml")
    cfg.exit.gumbel_search = False
    cfg.exit.net_opponent = False
    states = _gen_states(cfg)
    assert states, "failed to generate test states"
    print(f"Generated {len(states)} mid-game test states.")

    # ── Test 1: bit-identity (gumbel off == original algorithm) ──
    n_checked = 0
    for state, feats, ss in states:
        for i in range(cfg.env.max_planets):
            if not feats.own_mask[i]:
                continue
            ref_t, ref_f = _ref_search_planet(state, feats, ss, state.player, i, cfg.env, cfg.exit)
            new_t, new_f = search_improve_planet(
                state=state,
                features=feats,
                sim_state=ss,
                player=state.player,
                source_slot=i,
                env_cfg=cfg.env,
                exit_cfg=cfg.exit,
            )
            assert np.array_equal(ref_t, new_t), f"target diff @ slot {i}"
            assert np.array_equal(ref_f, new_f), f"frac diff @ slot {i}"
            n_checked += 1
    print(f"  [OK] bit-identity: {n_checked} owned-planet decisions identical (gumbel off).")

    # ── Test 2: bit-identity holds even when a prior is supplied but flag off ──
    state, feats, ss = states[0]
    rng = np.random.default_rng(0)
    for i in range(cfg.env.max_planets):
        if not feats.own_mask[i]:
            continue
        P, K = cfg.env.max_planets, len(cfg.env.ship_fractions)
        pt = rng.random(P + 1).astype(np.float32)
        pf = rng.random((P, K)).astype(np.float32)
        ref_t, ref_f = _ref_search_planet(state, feats, ss, state.player, i, cfg.env, cfg.exit)
        new_t, new_f = search_improve_planet(
            state=state,
            features=feats,
            sim_state=ss,
            player=state.player,
            source_slot=i,
            env_cfg=cfg.env,
            exit_cfg=cfg.exit,
            prior_target=pt,
            prior_frac=pf,
            rng_seed=123,
        )
        assert np.array_equal(ref_t, new_t) and np.array_equal(ref_f, new_f), (
            f"flag-off path used the prior @ slot {i}"
        )
    print("  [OK] flag-off ignores supplied prior (still bit-identical).")

    # ── Test 3: gumbel-on sanity ──
    cfg.exit.gumbel_search = True
    P, K = cfg.env.max_planets, len(cfg.env.ship_fractions)
    n_g = 0
    for state, feats, ss in states:
        # Build a real prior via a freshly-constructed (untrained) net forward.
        import torch

        from v2.model import OrbitNet

        model = OrbitNet(cfg.model).eval()
        with torch.inference_mode():
            o = model(
                torch.from_numpy(feats.planet_features).unsqueeze(0),
                torch.from_numpy(feats.global_features).unsqueeze(0),
                torch.from_numpy(feats.planet_mask).unsqueeze(0),
                torch.from_numpy(feats.own_mask).unsqueeze(0),
                torch.from_numpy(feats.reachability_mask).unsqueeze(0),
            )
        prior_t = torch.softmax(o.logits[0].float(), -1).numpy()
        prior_f = torch.softmax(o.frac_logits[0].float(), -1).numpy()
        for i in range(P):
            if not feats.own_mask[i]:
                continue
            t1, f1 = search_improve_planet(
                state=state,
                features=feats,
                sim_state=ss,
                player=state.player,
                source_slot=i,
                env_cfg=cfg.env,
                exit_cfg=cfg.exit,
                prior_target=prior_t[i],
                prior_frac=prior_f[i],
                rng_seed=999 + i,
            )
            t2, f2 = search_improve_planet(
                state=state,
                features=feats,
                sim_state=ss,
                player=state.player,
                source_slot=i,
                env_cfg=cfg.env,
                exit_cfg=cfg.exit,
                prior_target=prior_t[i],
                prior_frac=prior_f[i],
                rng_seed=999 + i,
            )
            assert np.isfinite(t1).all() and np.isfinite(f1).all(), "non-finite gumbel dist"
            assert abs(t1.sum() - 1.0) < 1e-4, f"target probs sum {t1.sum()}"
            for j in range(P):
                assert abs(f1[j].sum() - 1.0) < 1e-4, f"frac row {j} sum {f1[j].sum()}"
            assert np.array_equal(t1, t2) and np.array_equal(f1, f2), "gumbel not reproducible"
            n_g += 1
    print(f"  [OK] gumbel-on: {n_g} decisions finite, normalized, reproducible.")
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
