"""Self-consistency check for divergence_mine.py.

Premise of the mine: feed the SAME recorded obs sequence through a fresh
`producer` runtime and capture what it WOULD launch. If that counterfactual is
faithful, then running it against a seat that was ACTUALLY played by `producer`
must reproduce that seat's real launches almost exactly:
  Jaccard(actual.pairs, cf_producer.pairs) ~ 1.0  and  wave-delta ~ 0.

If instead we see the same low-Jaccard / big-negative-wave-delta pattern the mine
reports vs the top clones, the mine's counterfactual is the artifact (not signal).

Run:  uv run python scripts/divergence_mine_selfcheck.py --opp distance_1100 --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.divergence_mine import analyze_seat, jaccard  # noqa: E402


def play_replay(opp: str, seed: int) -> dict:
    """producer at seat 0 vs `opp` at seat 1 → replay dict in the mine's schema."""
    from kaggle_environments import make

    from agents import load_named_agent

    env = make("orbit_wars", configuration={"randomSeed": seed})
    env.run([load_named_agent("producer"), load_named_agent(opp)])
    # toJSON gives plain json-serializable dicts (no Struct objects)
    blob = json.loads(json.dumps(env.toJSON()))
    blob.setdefault("info", {})["TeamNames"] = ["producer", opp]
    return blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opp", default="distance_1100")
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    all_jacc, all_wave_d, all_src_d, all_ship_d = [], [], [], []
    n_turns = 0
    n_shared_total = 0     # shared-source decisions (producer vs cf-producer)
    n_disagree_total = 0   # of those, where the dominant target differs
    for seed in seeds:
        rep = play_replay(args.opp, seed)
        steps = rep["steps"]
        n_players = len(steps[0])
        recs = analyze_seat(steps, seat=0, n_players=n_players)  # seat 0 = real producer
        for r in recs:
            a, p = r["a"], r["p"]  # actual producer seat vs cf producer
            n_turns += 1
            if a.n_launches or p.n_launches:
                all_jacc.append(jaccard(a.pairs, p.pairs))
            all_wave_d.append(a.n_launches - p.n_launches)
            all_src_d.append(a.n_sources - p.n_sources)
            all_ship_d.append(a.total_ships - p.total_ships)
            n_shared_total += r["n_shared"]
            n_disagree_total += len(r["tgt_disagree"])
        print(f"  seed {seed}: {len(recs)} producer-seat turns")

    def stats(name, xs):
        if not xs:
            print(f"  {name:18s}: (no data)")
            return
        print(f"  {name:18s}: mean={st.mean(xs):+.3f}  median={st.median(xs):+.3f}  "
              f"n={len(xs)}  min={min(xs):+.2f} max={max(xs):+.2f}")

    print(f"\n=== Self-consistency: real producer seat vs cf_producer ({n_turns} turns) ===")
    stats("Jaccard(actual,cf)", all_jacc)
    stats("wave delta", all_wave_d)
    stats("sources delta", all_src_d)
    stats("ships delta", [float(x) for x in all_ship_d])
    j = st.mean(all_jacc) if all_jacc else float("nan")
    wd = st.mean(all_wave_d) if all_wave_d else float("nan")
    print("\nVERDICT:")
    if j >= 0.9 and abs(wd) <= 0.3:
        print(f"  FAITHFUL — Jaccard {j:.2f}, wave delta {wd:+.2f}. The mine reproduces a real"
              "\n  producer seat → the -73% wave / 0.1 Jaccard vs top clones is REAL signal.")
    else:
        print(f"  SUSPECT — Jaccard {j:.2f}, wave delta {wd:+.2f}. The mine does NOT reproduce"
              "\n  a real producer seat → the divergence table is contaminated by a harness bug.")

    # --- Target-disagreement profiler self-check ---------------------------------
    # Producer is deterministic: feeding its own recorded obs back through a fresh
    # producer must pick the SAME dominant target on every shared source. So the
    # disagreement profiler must surface ZERO disagreements on a real producer seat.
    # Any non-zero count = a bug in shared-source / dominant-target matching.
    print(f"\n=== Target-disagreement self-check: {n_disagree_total} disagreements over "
          f"{n_shared_total} shared-source decisions ===")
    if n_disagree_total == 0:
        print("  CLEAN — producer never disagrees with itself on shared sources → the"
              "\n  shared-source/dominant-target matching is sound. Profiler output is trustworthy.")
    else:
        print(f"  BUG — {n_disagree_total} spurious self-disagreements. The shared-source or"
              "\n  dominant-target matching is broken; FIX before trusting the profile table.")
        sys.exit(1)


if __name__ == "__main__":
    main()
