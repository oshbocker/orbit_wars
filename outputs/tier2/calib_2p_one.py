"""Collect 2P ENSEMBLE detector precision sequences for ONE opponent.

Seat 0 = our v5 (contest_ensemble=1, contest_detect=1; runs BOTH producer models —
base β=0 and producer_v2 β=2.2 — predicts seat 1 every turn, verifies next turn).
Seat 1 = <opponent>. _DETECT_DEBUG records (step, seat, model, n_pred, n_obs, n_inter)
per measured turn; precision = n_inter / n_pred. Saves /tmp/calib2p_<opp>.npy = a list
of games, each a list of (step, {model: precision}) ordered by step.

Usage: python calib_2p_one.py <opponent> <games> [seed0]
"""
from __future__ import annotations

import dataclasses
import importlib.util
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ["OMP_NUM_THREADS"] = "1"
ROOT = Path("/home/aeschbacher/git/orbit_wars")
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

torch.set_num_threads(1)
from agents import _normalize_obs, load_named_agent  # noqa: E402

_counter = 0


def build_v5_detect(dbg):
    global _counter
    _counter += 1
    main_py = ROOT / "agents" / "v5" / "main.py"
    modname = f"_v5c_{_counter}"
    spec = importlib.util.spec_from_file_location(modname, main_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    cfg2 = dataclasses.replace(mod.ProducerLiteConfig(), contest_waves=2, contest_detect=1, contest_ensemble=1)
    mod._config_for = lambda pc: (cfg2 if int(pc) < 4 else mod.CONFIG_4P)
    mod._DETECT_DEBUG = dbg
    fn = mod.agent
    return lambda obs, config=None: fn(_normalize_obs(obs))


def main():
    opponent = sys.argv[1]
    games = int(sys.argv[2])
    seed0 = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    from kaggle_environments import make
    out_games = []
    for g in range(games):
        dbg: list = []
        v5 = build_v5_detect(dbg)
        env = make("orbit_wars", configuration={"randomSeed": seed0 + g})
        env.run([v5, load_named_agent(opponent)])
        # group by step (seat is always 1 in 2P): {step: {model: precision}}
        per_step: dict[int, dict[int, float]] = defaultdict(dict)
        for (step, seat, model, n_pred, n_obs, n_inter) in dbg:
            if n_pred > 0:
                per_step[step][model] = n_inter / n_pred
        seq = [(s, per_step[s]) for s in sorted(per_step)]
        out_games.append(seq)
        npts = sum(len(s[1]) for game in out_games for s in game)
        print(f"[{opponent}] game {g+1}/{games} done, {len(out_games)} games {npts} model-obs", flush=True)
    np.save(f"/tmp/calib2p_{opponent}.npy", np.array(out_games, dtype=object), allow_pickle=True)
    print(f"[{opponent}] DONE -> /tmp/calib2p_{opponent}.npy", flush=True)


if __name__ == "__main__":
    main()
