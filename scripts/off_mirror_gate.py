"""Off-mirror opponent-injection gate (the missing instrument from Cluster 11).

Cluster 11 tested v5's opponent-injection hook (``opp_inject_waves``) only as a MIRROR
A/B (``v5:opp_inject_waves=N`` vs ``v5``) and it went INERT (+1-2pp over a 55.4% A/A
floor). Root cause: the mirror can only measure level-1-vs-level-0 exploitability of a
*producer-self-model*, and the flow-diff scorer already prices producer-like opponents
in. Neither the mirror nor the public pool (which ceilings at ~99% vs producer-tier) can
see whether anticipating a genuinely NON-producer opponent buys anything.

This gate builds that instrument: inject-ON vs inject-OFF, BOTH playing AGAINST the
non-producer archetype fixtures (``half_drainer`` = Isaiah-style partial-send;
``swarmer`` = 213tubo-style many-small-waves), side-alternated on paired seeds. Because
producer-tier v5 dominates the hand-built fixtures on win-rate (~100% — the same ceiling
problem), the PRIMARY instrument is MARGIN (steps-to-decision and final ship-score
differential), exactly the signal Cluster 11 flagged as the only one it could see. The
comparison is paired by seed (same map for ON and OFF), so small systematic margin
shifts are detectable.

    # run the dose curve vs both archetypes (one resumable CSV per archetype)
    uv run python scripts/off_mirror_gate.py \
        --archetypes half_drainer,swarmer --doses 1,3,6 --games 120 --workers 6

    # re-print the analysis without replaying
    uv run python scripts/off_mirror_gate.py --analyze \
        --archetypes half_drainer,swarmer --doses 1,3,6

Output: ``outputs/arena/offmirror_<archetype>.csv`` (dose 0 = inject-OFF baseline +
each requested dose). Resumable: bump ``--games`` to extend.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.arena import _build_agent, _final_scores, _worker_init  # noqa: E402

FIELDS = [
    "archetype",
    "dose",
    "seed",
    "v5_side",
    "outcome",  # "v5" | "arch" | "tie"
    "reward_v5",
    "reward_arch",
    "score_v5",  # final ship-score (planets + fleets) for v5
    "score_arch",
    "steps",
    "status_v5",
    "status_arch",
    "seconds",
]


def _variant_spec(dose: int) -> str:
    return "v5" if int(dose) == 0 else f"v5:opp_inject_waves={int(dose)}"


def _play(job: tuple[str, int, int, int]) -> dict:
    """One game: (archetype, dose, seed, v5_side). v5 variant vs the archetype."""
    archetype, dose, seed, v5_side = job
    from kaggle_environments import make

    t0 = time.time()
    agents: list = [None, None]
    agents[v5_side] = _build_agent(_variant_spec(dose))
    agents[1 - v5_side] = _build_agent(archetype)
    env = make("orbit_wars", configuration={"randomSeed": seed})
    env.run(agents)
    last = env.steps[-1]
    rv = last[v5_side].reward
    ra = last[1 - v5_side].reward
    rv = -1 if rv is None else rv
    ra = -1 if ra is None else ra
    scores = _final_scores(env, 2)
    outcome = "tie" if rv == ra else ("v5" if rv > ra else "arch")
    return {
        "archetype": archetype,
        "dose": int(dose),
        "seed": seed,
        "v5_side": v5_side,
        "outcome": outcome,
        "reward_v5": rv,
        "reward_arch": ra,
        "score_v5": round(scores[v5_side], 1),
        "score_arch": round(scores[1 - v5_side], 1),
        "steps": len(env.steps) - 1,
        "status_v5": last[v5_side].status,
        "status_arch": last[1 - v5_side].status,
        "seconds": round(time.time() - t0, 1),
    }


def _out_csv(archetype: str) -> Path:
    return ROOT / "outputs" / "arena" / f"offmirror_{archetype}.csv"


def _load_done(csv_path: Path) -> set[tuple[int, int, int]]:
    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add((int(row["dose"]), int(row["seed"]), int(row["v5_side"])))
    return done


# ---- analysis ---------------------------------------------------------------


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (p, lo, hi)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return p, max(0.0, center - half), min(1.0, center + half)


def _mean_sd(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def _rows(archetype: str) -> list[dict]:
    csv_path = _out_csv(archetype)
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def analyze(archetype: str, doses: list[int]) -> None:
    rows = _rows(archetype)
    if not rows:
        print(f"\n[{archetype}] no data at {_out_csv(archetype)}")
        return
    by_dose: dict[int, list[dict]] = {}
    for r in rows:
        by_dose.setdefault(int(r["dose"]), []).append(r)

    print(f"\n=== off-mirror gate: v5 vs {archetype} ===")
    print(f"    {'cell':<14}{'n':>4}{'v5 win%':>10}{'95% CI':>16}"
          f"{'steps':>9}{'score_v5':>10}{'score_arch':>11}{'margin':>9}")

    base = by_dose.get(0, [])

    def cell_stats(rs: list[dict]):
        n = len(rs)
        wins = sum(1 for r in rs if r["outcome"] == "v5")
        p, lo, hi = _wilson(wins, n)
        steps = [float(r["steps"]) for r in rs]
        sv = [float(r["score_v5"]) for r in rs]
        sa = [float(r["score_arch"]) for r in rs]
        margin = [float(r["score_v5"]) - float(r["score_arch"]) for r in rs]
        return n, p, lo, hi, _mean_sd(steps)[0], _mean_sd(sv)[0], _mean_sd(sa)[0], _mean_sd(margin)[0]

    for dose in [0, *doses]:
        rs = by_dose.get(dose, [])
        if not rs:
            continue
        label = "inject-OFF" if dose == 0 else f"waves={dose}"
        n, p, lo, hi, st, sv, sa, mg = cell_stats(rs)
        print(f"    {label:<14}{n:>4}{p:>9.1%}{f'[{lo:.0%},{hi:.0%}]':>16}"
              f"{st:>9.1f}{sv:>10.1f}{sa:>11.1f}{mg:>9.1f}")

    # Paired (same-seed) margin deltas: inject-ON minus inject-OFF. This is the
    # sensitive instrument — same map, so it removes per-map variance.
    if base:
        base_by_key = {(int(r["seed"]), int(r["v5_side"])): r for r in base}
        print("\n    paired Δ vs inject-OFF (same seed+side; +Δ favors injection):")
        for dose in doses:
            rs = by_dose.get(dose, [])
            d_steps: list[float] = []
            d_margin: list[float] = []
            d_win: list[float] = []
            for r in rs:
                b = base_by_key.get((int(r["seed"]), int(r["v5_side"])))
                if b is None:
                    continue
                # fewer steps = faster close-out = good => negate so +Δ = better
                d_steps.append(float(-(int(r["steps"]) - int(b["steps"]))))
                d_margin.append(
                    (float(r["score_v5"]) - float(r["score_arch"]))
                    - (float(b["score_v5"]) - float(b["score_arch"]))
                )
                d_win.append(float((1 if r["outcome"] == "v5" else 0) - (1 if b["outcome"] == "v5" else 0)))
            if not d_steps:
                continue
            ms, ss = _mean_sd(d_steps)
            mm, sm = _mean_sd(d_margin)
            se_s = ss / math.sqrt(len(d_steps)) if d_steps else 0.0
            se_m = sm / math.sqrt(len(d_margin)) if d_margin else 0.0
            mw = sum(d_win) / len(d_win) if d_win else 0.0
            sig_s = "  *" if abs(ms) > 1.96 * se_s and se_s > 0 else ""
            sig_m = "  *" if abs(mm) > 1.96 * se_m and se_m > 0 else ""
            print(
                f"      waves={dose}  n={len(d_steps):>3}  "
                f"Δsteps_faster={ms:+6.1f} (±{1.96 * se_s:4.1f}){sig_s:<3}  "
                f"Δscore_margin={mm:+7.1f} (±{1.96 * se_m:5.1f}){sig_m:<3}  "
                f"Δwin%={mw:+.1%}"
            )
        print("    (* = paired mean is >1.96 SE from 0; 'faster' negates steps so +Δ = better)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archetypes", default="half_drainer,swarmer")
    ap.add_argument("--doses", default="1,3,6", help="opp_inject_waves doses (0/OFF always included)")
    ap.add_argument("--games", type=int, default=120, help="paired games per cell (>=120 = project floor)")
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--analyze", action="store_true", help="only print analysis from existing CSVs")
    args = ap.parse_args()

    archetypes = [a.strip() for a in args.archetypes.split(",") if a.strip()]
    doses = [int(d) for d in args.doses.split(",") if d.strip()]

    if not args.analyze:
        for archetype in archetypes:
            csv_path = _out_csv(archetype)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            done = _load_done(csv_path)
            jobs = []
            for dose in [0, *doses]:
                for i in range(args.games):
                    seed, v5_side = args.seed + i, i % 2
                    if (dose, seed, v5_side) not in done:
                        jobs.append((archetype, dose, seed, v5_side))
            print(f"[{archetype}] {len(jobs)} games to play ({len(done)} already in {csv_path})")
            if not jobs:
                continue
            new_file = not csv_path.exists()
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                if new_file:
                    writer.writeheader()
                from multiprocessing import Pool

                t0 = time.time()
                with Pool(args.workers, initializer=_worker_init, maxtasksperchild=4) as pool:
                    for k, row in enumerate(pool.imap_unordered(_play, jobs), 1):
                        writer.writerow(row)
                        f.flush()
                        err = " ⚠" if "ERROR" in (row["status_v5"], row["status_arch"]) else ""
                        print(
                            f"  [{k}/{len(jobs)}] {archetype} dose={row['dose']} "
                            f"seed={row['seed']} side={row['v5_side']} -> {row['outcome']} "
                            f"(sv={row['score_v5']} sa={row['score_arch']} st={row['steps']}, "
                            f"{row['seconds']}s){err}",
                            flush=True,
                        )
                print(f"  done in {(time.time() - t0) / 60:.1f} min")

    for archetype in archetypes:
        analyze(archetype, doses)


if __name__ == "__main__":
    main()
