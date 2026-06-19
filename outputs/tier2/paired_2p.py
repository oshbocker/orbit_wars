"""Paired 2P gate: contest-ENSEMBLE (contest_ensemble=1) vs contest-SINGLE
(contest_ensemble=0, == v5.4) on IDENTICAL boards. For each seed + opponent, play
our seat with each variant, side-alternated (our agent as seat 0 AND seat 1). Same
board + opponent => isolates the ensemble. Writes outputs/arena/paired_2p.csv.

Usage: python paired_2p.py <opp1,opp2,...> <n_seeds> [seed0]
Win = our reward > opp reward (ties excluded from win count, reported separately).
"""
from __future__ import annotations

import csv
import dataclasses
import importlib.util
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
ROOT = Path("/home/aeschbacher/git/orbit_wars")
sys.path.insert(0, str(ROOT))

VARIANTS = [
    ("single", dict(contest_waves=2, contest_ensemble=0)),
    ("ens", dict(contest_waves=2, contest_ensemble=1)),
]


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)


def build_v5(overrides, tag):
    main_py = ROOT / "agents" / "v5" / "main.py"
    spec = importlib.util.spec_from_file_location(f"_v5p_{tag}", main_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_v5p_{tag}"] = mod
    spec.loader.exec_module(mod)
    cfg = dataclasses.replace(mod.ProducerLiteConfig(), **overrides)
    mod._config_for = lambda pc: (cfg if int(pc) < 4 else mod.CONFIG_4P)
    fn = mod.agent
    from agents import _normalize_obs
    return lambda obs, config=None: fn(_normalize_obs(obs))


def _scores(env):
    obs = env.steps[-1][0].observation
    sc = [0.0, 0.0]
    for p in (obs["planets"] if isinstance(obs, dict) else obs.planets) or []:
        o = int(p[1])
        if 0 <= o < 2:
            sc[o] += float(p[5])
    for fl in (obs["fleets"] if isinstance(obs, dict) else obs.fleets) or []:
        o = int(fl[1])
        if 0 <= o < 2:
            sc[o] += float(fl[6])
    return sc


def play(job):
    opp, seed, side, vlabel, ov = job
    from agents import load_named_agent
    from kaggle_environments import make
    us = build_v5(ov, f"{seed}_{side}_{vlabel}")
    them = load_named_agent(opp)
    agents = [None, None]
    agents[side] = us
    agents[1 - side] = them
    env = make("orbit_wars", configuration={"randomSeed": seed})
    env.run(agents)
    last = env.steps[-1]
    ru = -1.0 if last[side].reward is None else float(last[side].reward)
    rt = -1.0 if last[1 - side].reward is None else float(last[1 - side].reward)
    sc = _scores(env)
    margin = sc[side] - sc[1 - side]
    win = 1 if ru > rt else 0
    tie = 1 if ru == rt else 0
    return [opp, seed, side, vlabel, win, tie, round(ru, 1), round(margin, 1), len(env.steps) - 1]


def main():
    opps = sys.argv[1].split(",")
    n = int(sys.argv[2])
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 40000
    from multiprocessing import Pool
    jobs = [
        (opp, seed0 + i, side, vl, ov)
        for opp in opps
        for i in range(n)
        for side in (0, 1)
        for vl, ov in VARIANTS
    ]
    out = ROOT / "outputs" / "arena" / "paired_2p.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["opp", "seed", "side", "variant", "win", "tie", "reward", "margin", "steps"])
        with Pool(7, initializer=_init, maxtasksperchild=4) as pool:
            for k, row in enumerate(pool.imap_unordered(play, jobs), 1):
                w.writerow(row)
                f.flush()
                rows.append(row)
                if k % 20 == 0:
                    print(f"[{k}/{len(jobs)}] ...", flush=True)
    # summary
    print("\n=== paired 2P summary (per opp, per variant) ===")
    print(f"{'opp':16s} {'var':7s}  n   win%   tie%   meanRew  meanMargin")
    agg = {}
    for r in rows:
        opp, _s, _side, vl, win, tie, rew, margin, _steps = r
        k = (opp, vl)
        a = agg.setdefault(k, [0, 0, 0, 0.0, 0.0])
        a[0] += 1; a[1] += win; a[2] += tie; a[3] += rew; a[4] += margin
    for opp in opps:
        for vl, _ in VARIANTS:
            a = agg.get((opp, vl))
            if not a:
                continue
            nn = a[0]
            print(f"{opp:16s} {vl:7s} {nn:3d}  {100*a[1]/nn:5.1f}  {100*a[2]/nn:5.1f}  "
                  f"{a[3]/nn:+7.2f}  {a[4]/nn:+9.1f}")
    print("\n=== ENS - SINGLE deltas (paired) ===")
    for opp in opps:
        s = agg.get((opp, "single")); e = agg.get((opp, "ens"))
        if not s or not e:
            continue
        dwin = 100*(e[1]/e[0] - s[1]/s[0])
        dmar = e[4]/e[0] - s[4]/s[0]
        print(f"{opp:16s}  win {dwin:+5.1f}   margin {dmar:+8.1f}")


if __name__ == "__main__":
    main()
