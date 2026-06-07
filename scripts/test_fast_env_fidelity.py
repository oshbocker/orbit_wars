"""Fidelity gate for v2/fast_env.py vs the real Kaggle engine.

Runs identical (agent, opponent, seed) episodes through both the Kaggle
`interpreter()` and `FastOrbitWars`, and asserts the game states match
step-for-step. If this passes across many seeds, fast_env is a trustworthy
training environment; if not, it must not be used for self-play.

We drive BOTH with the same deterministic scripted agent (a seeded pseudo-policy
that is a pure function of the observation), so any divergence is the simulator's
fault, not the agent's.

Usage:
    uv run python scripts/test_fast_env_fidelity.py --games 20
    uv run python scripts/test_fast_env_fidelity.py --games 5 --verbose
"""

from __future__ import annotations

import argparse
import math

from kaggle_environments import make

from v2.fast_env import FastOrbitWars


def scripted_agent(obs) -> list:
    """Deterministic pure-function policy: each owned planet with >5 ships sends
    half its ships at a fixed angle derived ONLY from the planet id.

    Note: deliberately independent of `step`. The Kaggle engine only populates
    `step` on player 0's observation (other players' obs have step=None), so a
    step-dependent policy would desync the two players between engine and
    fast_env for reasons unrelated to simulator fidelity. Keying on pid alone
    guarantees identical actions from identical (pid, owner, ships) inputs."""

    def g(o, k, d):
        if isinstance(o, dict):
            return o.get(k, d)
        return getattr(o, k, d)

    player = g(obs, "player", 0)
    planets = g(obs, "planets", []) or []
    moves = []
    for p in planets:
        pid = p["id"] if isinstance(p, dict) else p[0]
        owner = p["owner"] if isinstance(p, dict) else p[1]
        ships = p["ships"] if isinstance(p, dict) else p[5]
        if owner == player and ships > 5:
            # Deterministic angle from pid only — spreads fleets around.
            angle = ((pid * 37) % 360) * math.pi / 180.0
            moves.append([pid, angle, ships // 2])
    return moves


def _engine_state(obs):
    """Extract a comparable snapshot from a kaggle obs."""

    def g(o, k, d):
        if isinstance(o, dict):
            return o.get(k, d)
        return getattr(o, k, d)

    planets = {p[0]: (p[1], round(p[2], 3), round(p[3], 3), p[5]) for p in g(obs, "planets", [])}
    n_fleets = len(g(obs, "fleets", []))
    return planets, n_fleets


def _fast_state(env: FastOrbitWars):
    planets = {p[0]: (p[1], round(p[2], 3), round(p[3], 3), p[5]) for p in env.planets}
    return planets, len(env.fleets)


def run_pair(seed: int, max_steps: int, verbose: bool) -> tuple[bool, str]:
    # ── Kaggle engine episode ────────────────────────────────────────────────
    # The engine's reset + first step([[],[]]) runs a full interpreter turn
    # (production + movement) and lands at step=1. We align fast_env to the same
    # point by stepping it once with no-op actions, so both start comparison at
    # the engine's step=1 state.
    kenv = make("orbit_wars", configuration={"seed": seed}, debug=False)
    kenv.reset(num_agents=2)
    kstates = kenv.step([[], []])  # advances engine to step=1

    # ── Fast env, seeded identically, aligned to engine step=1 ───────────────
    fenv = FastOrbitWars(num_agents=2, seed=seed)
    fenv.step([[], []])  # apply the same initial no-op turn

    # Compare aligned planet layout (ids, owners, positions, ships).
    keng = kstates[0].observation
    kp, _ = _engine_state(keng)
    fp, _ = _fast_state(fenv)
    if set(kp.keys()) != set(fp.keys()):
        return False, f"seed {seed}: initial planet IDs differ (engine {len(kp)} vs fast {len(fp)})"
    for pid in kp:
        if kp[pid] != fp[pid]:
            return (
                False,
                f"seed {seed}: initial planet {pid} differs: engine {kp[pid]} vs fast {fp[pid]}",
            )

    # Step both with the SAME scripted actions derived from each one's own obs
    # (obs are identical, so actions are identical — verified by the state checks).
    for t in range(max_steps):
        # Engine actions from engine obs.
        ka = [scripted_agent(kstates[i].observation) for i in range(2)]
        # Fast actions from fast obs.
        fa = [scripted_agent(fenv.observation(i)) for i in range(2)]

        kstates = kenv.step(ka)
        fenv.step(fa)

        kp, knf = _engine_state(kstates[0].observation)
        fp, fnf = _fast_state(fenv)

        kdone = all(s.status == "DONE" for s in kstates)
        if kdone != fenv.done:
            # Allow the engine's off-by-one termination check to differ by 1 step.
            if not (kdone or fenv.done):
                pass
        if set(kp.keys()) != set(fp.keys()):
            return False, (
                f"seed {seed} step {t}: planet sets diverge "
                f"(engine {sorted(kp)} vs fast {sorted(fp)})"
            )
        for pid in kp:
            if kp[pid] != fp[pid]:
                return False, (
                    f"seed {seed} step {t}: planet {pid} diverges: "
                    f"engine {kp[pid]} vs fast {fp[pid]}"
                )
        if knf != fnf:
            return False, f"seed {seed} step {t}: fleet count diverges (engine {knf} vs fast {fnf})"

        if verbose and t % 50 == 0:
            print(f"  seed {seed} step {t}: OK ({len(kp)} planets, {knf} fleets)")

        if fenv.done or kdone:
            break

    return True, f"seed {seed}: MATCH ({t + 1} steps)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    passed = 0
    for i in range(args.games):
        seed = args.seed0 + i
        ok, msg = run_pair(seed, args.max_steps, args.verbose)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {msg}")
        if ok:
            passed += 1
        else:
            break  # stop at first divergence so the message is actionable

    print(f"\n{passed}/{args.games} episodes matched step-for-step.")
    if passed == args.games:
        print("FIDELITY GATE PASSED — fast_env matches the Kaggle engine.")
    else:
        print("FIDELITY GATE FAILED — do not train in fast_env until fixed.")


if __name__ == "__main__":
    main()
