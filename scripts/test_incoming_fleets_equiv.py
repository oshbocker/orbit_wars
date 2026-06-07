"""Equivalence + benchmark harness for the vectorized compute_incoming_fleets.

Drives real apex-vs-apex games (2p and 4p, many seeds, all game phases) and, at
every step, compares the vectorized `compute_incoming_fleets` against the scalar
reference `_compute_incoming_fleets_scalar` for every player perspective. Asserts
identical target planets and per-team ship/ETA aggregates (within tolerance), then
reports the wall-clock speedup.

Run: uv run python scripts/test_incoming_fleets_equiv.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.apex import agent as apex
from src.game_types import parse_observation
from v2.fast_env import FastOrbitWars
from v2.state import _compute_incoming_fleets_scalar, compute_incoming_fleets

SHIP_TOL = 1e-6
ETA_TOL = 1e-4


def _rng_agent_factory(seed: int):
    """Fast random-launch agent: fires fleets in varied directions (broad
    geometry coverage — sun crossings, OOB, orbiting hits) without apex's cost."""
    import math
    import random

    rng = random.Random(seed)

    def act(obs):
        planets = obs["planets"]
        player = obs["player"]
        moves = []
        for p in planets:
            if p["owner"] == player and p["ships"] > 1 and rng.random() < 0.2:
                ang = rng.uniform(0, 2 * math.pi)
                moves.append([p["id"], ang, max(1, int(p["ships"] * rng.random()))])
        return moves

    return act


def _compare(a: dict, b: dict, ctx: str, mism: list) -> None:
    """Compare two {planet_id: IncomingFleetInfo} dicts."""
    if set(a) != set(b):
        mism.append(f"{ctx}: target sets differ scalar={sorted(a)} vec={sorted(b)}")
        return
    for pid in a:
        ia, ib = a[pid], b[pid]
        for team in range(4):
            if abs(ia.ships[team] - ib.ships[team]) > SHIP_TOL:
                mism.append(
                    f"{ctx} pid={pid} team={team}: ships {ia.ships[team]} vs {ib.ships[team]}"
                )
            # ETA only meaningful where ships present.
            if ia.ships[team] > 0 and abs(ia.eta[team] - ib.eta[team]) > ETA_TOL:
                mism.append(f"{ctx} pid={pid} team={team}: eta {ia.eta[team]} vs {ib.eta[team]}")


def main() -> int:
    mism: list[str] = []
    n_states = 0
    n_fleets_total = 0
    t_scalar = 0.0
    t_vec = 0.0
    max_fleets = 0

    # Bulk coverage via fast random-launch agents; small apex sample for realism.
    configs = [
        (2, range(0, 8), "rng"),
        (4, range(100, 105), "rng"),
        (2, range(900, 902), "apex"),
        (4, range(950, 951), "apex"),
    ]
    for num_agents, seeds, kind in configs:
        for seed in seeds:
            sim = FastOrbitWars(num_agents=num_agents, seed=seed)
            agents = (
                [apex] * num_agents
                if kind == "apex"
                else [_rng_agent_factory(seed * 17 + p) for p in range(num_agents)]
            )
            for _ in range(160):
                if sim.done:
                    break
                acts = [agents[p](sim.observation(p)) for p in range(num_agents)]
                sim.step(acts)
                for p in range(num_agents):
                    st = parse_observation(sim.observation(p))
                    n_states += 1
                    n_fleets_total += len(st.fleets)
                    max_fleets = max(max_fleets, len(st.fleets))

                    t0 = time.perf_counter()
                    ref = _compute_incoming_fleets_scalar(st, st.player)
                    t_scalar += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    vec = compute_incoming_fleets(st, st.player)
                    t_vec += time.perf_counter() - t0

                    _compare(ref, vec, f"a{num_agents}/s{seed}/p{p}/step{st.step}", mism)

    print(f"states compared : {n_states}")
    print(f"total fleets     : {n_fleets_total} (max in a state: {max_fleets})")
    print(f"scalar time      : {t_scalar * 1000:.0f}ms  ({t_scalar / n_states * 1000:.3f}ms/call)")
    print(f"vectorized time  : {t_vec * 1000:.0f}ms  ({t_vec / n_states * 1000:.3f}ms/call)")
    print(f"speedup          : {t_scalar / max(t_vec, 1e-9):.1f}x")
    print(f"mismatches       : {len(mism)}")
    for m in mism[:25]:
        print("  " + m)
    return 1 if mism else 0


if __name__ == "__main__":
    sys.exit(main())
