"""Step-1 gating probe: does a STRONGER/DIFFERENT agent's attacks fall inside producer's
candidate grid? If coverage is high, the candidate-restricted RichSelector can clone it
(picking a different candidate than producer's argmax = the learnable delta). If low, we
must broaden producer's candidate generation BEFORE harvesting top-tier replays.

Proxies for the 1500-1700 tier (not vendored): tamrazov_1224 (different lineage),
producer_v2 (reinforce-risk, beats producer), v5 (our fork).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.macro_relabel import posture_of, resolve_launches_for_step  # noqa: E402
from scripts.producer_features import MAXP, ProducerFeatureExtractor  # noqa: E402


def _obs(e):
    return e["observation"] if not hasattr(e, "observation") else e.observation


def probe(agent_name, games, seed):
    from kaggle_environments import make

    from agents import load_named_agent

    ext = ProducerFeatureExtractor()
    n_atk = in_grid = argmax_match = 0
    for g in range(games):
        env = make("orbit_wars", configuration={"randomSeed": seed + g})
        # seat 0 = the agent we want to clone; seat 1 = producer (just an opponent)
        env.run([load_named_agent(agent_name), load_named_agent("producer")])
        steps = env.steps
        for t in range(1, len(steps), 4):  # subsample for speed
            mls = [m for m in resolve_launches_for_step(steps, t, 0)
                   if m.dst_id >= 0 and posture_of(m) != "reinforce"]
            if not mls:
                continue
            obs = _obs(steps[t - 1][0])
            grid = ext.extract(obs)
            valid, score = grid["valid"], grid["score"]
            for ml in mls:
                i, j = ml.src_id, ml.dst_id
                if not (0 <= i < MAXP and 0 <= j < MAXP):
                    continue
                n_atk += 1
                if bool(valid[i, j]):
                    in_grid += 1
                    row = score[i]
                    if bool(torch.isfinite(row).any()) and int(row.argmax()) == j:
                        argmax_match += 1
        print(f"  {agent_name} game {g} done", flush=True)
    cov = in_grid / max(n_atk, 1)
    am = argmax_match / max(n_atk, 1)
    print(f"\n=== {agent_name} attacks vs producer candidate grid ({games} games) ===")
    print(f"attack launches      : {n_atk}")
    print(f"IN producer grid     : {in_grid} ({cov:.1%})   <- coverage; need high to clone-up")
    print(f"== producer argmax   : {argmax_match} ({am:.1%})   <- agrees w/ producer (low = real delta to learn)")
    print(f"in-grid but NOT argmax: {in_grid - argmax_match} ({(cov-am):.1%})  <- the LEARNABLE selection delta")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", default="tamrazov_1224,producer_v2,v5")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20000)
    args = ap.parse_args()
    for a in args.agents.split(","):
        probe(a, args.games, args.seed)
