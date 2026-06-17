"""DAgger on-policy correction for the rich-representation selector (Track 1 escape hatch).

The single-pass BC selector regressed v5 (46% byte-identical prior -> 18% trained) despite
98.7% imitation. Two candidate causes: (1) compounding error at contested decisions,
(2) BC covariate shift — the net trained only on v5's on-policy states, then drifts into OOD
states at gate time. DAgger directly attacks (2): roll out the LEARNER (its own state
distribution), relabel those states with the EXPERT (v5), aggregate, retrain, iterate.

Decisive read after gating each iter vs v5:
  - parity recovers (~46%+)  => covariate shift WAS the cause; architecture viable, top-tier on.
  - stays ~18%               => the delta mechanism is fundamental; rich-BC-selection closed.

Relabeling is consistent with inference: v5's memory is rolled over the learner's exact obs
sequence (obs-driven projection => same Δnet grid the learner computed), and the labels are
v5's greedy-fired targets in those states. See rl_research/RICH_BC_SELECTION_FINDINGS.md.

    uv run python scripts/rich_dagger.py --init outputs/checkpoints/rich_bc_v5_fixed/ckpt40.pt \
        --base-cache outputs/macro_bc/rich_v5_fixed40.npz --iters 2 --games 24
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.producer_features import ProducerFeatureExtractor, load_v5_module  # noqa: E402
from scripts.rich_bc_train import N_PAIR, example_from, train_selector  # noqa: E402
from v2.config import load_v2_config  # noqa: E402


def collect_learner_obs(ckpt, config_path, games, seed):
    """Play learner-vs-v5 (seat-alternated) and return the learner's obs sequence per game.
    A FRESH learner + opponent per game so rolling memory resets each game."""
    from kaggle_environments import make

    from agents import load_named_agent
    from scripts.rich_bc_agent import build_rich_agent

    seqs = []
    for g in range(games):
        learner = build_rich_agent(ckpt, config_path)
        seq: list = []

        def rec(obs, config=None, _seq=seq, _ln=learner):
            _seq.append(obs)
            return _ln(obs)

        opp = load_named_agent("v5")
        env = make("orbit_wars", configuration={"randomSeed": seed + g})
        env.run([rec, opp] if g % 2 == 0 else [opp, rec])
        seqs.append(seq)
        print(f"  collect game {g}: learner saw {len(seq)} obs (seat {g % 2})", flush=True)
    return seqs


def relabel_with_v5(cfg, ext, P, obs_seq):
    """Roll a CLEAN v5 (no selector) over the learner's obs sequence; each step yields an
    example whose features = v5's (== learner's) rolling Δnet grid and whose label = v5's
    greedy-fired target. Obs-driven projection => grid matches what the learner saw."""
    mod = load_v5_module()
    out = []
    for obs in obs_seq:
        mod._FEATURE_SINK = {}
        try:
            mod.agent(obs)
        finally:
            sink, mod._FEATURE_SINK = mod._FEATURE_SINK, None
        if sink:
            out.append(example_from(cfg, ext, P, obs, sink))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v2_exit.yaml")
    ap.add_argument("--init", default="outputs/checkpoints/rich_bc_v5_fixed/ckpt40.pt",
                    help="starting (BC-trained) selector to roll out for iter 1")
    ap.add_argument("--base-cache", default="outputs/macro_bc/rich_v5_fixed40.npz",
                    help="BC dataset to aggregate DAgger examples onto")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--games", type=int, default=24, help="learner rollouts per iter")
    ap.add_argument("--seed", type=int, default=40000, help="collection seed (disjoint from gate 20000-119)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--outdir", default="outputs/checkpoints/rich_dagger_v5")
    args = ap.parse_args()

    cfg = load_v2_config(args.config)
    cfg.model.use_pair_features = True
    cfg.model.pair_feat_dim = N_PAIR
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P = cfg.env.max_planets
    ext = ProducerFeatureExtractor(max_planets=P)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    D = list(np.load(args.base_cache, allow_pickle=True)["ex"])
    print(f"base dataset: {len(D)} examples from {args.base_cache}")

    cur_ckpt = args.init
    for it in range(1, args.iters + 1):
        print(f"\n=== DAgger iter {it}: roll out {args.games} learner games ({cur_ckpt}) ===", flush=True)
        seqs = collect_learner_obs(cur_ckpt, args.config, args.games, args.seed + 1000 * it)
        new_ex = []
        for si, seq in enumerate(seqs):
            new_ex += relabel_with_v5(cfg, ext, P, seq)
            print(f"  relabel seq {si}: +{len(new_ex)} cumulative new examples", flush=True)
        D += new_ex
        print(f"iter {it}: aggregated D = {len(D)} examples (+{len(new_ex)} on-policy)", flush=True)

        # Fresh retrain on the aggregate (standard DAgger), best-val snapshot.
        best_state, best_acc = train_selector(cfg, D, dev, epochs=args.epochs, seed=it)
        out = outdir / f"ckpt_iter{it}.pt"
        torch.save({"model": best_state, "config": args.config, "pair_feat_dim": N_PAIR}, out)
        print(f"iter {it}: saved best-val (acc={best_acc:.3f}) -> {out}", flush=True)
        cur_ckpt = str(out)

    print(f"\nDONE. Gate with:\n  uv run python scripts/arena.py --agents "
          f"\"richbc:{cur_ckpt}:{args.config},v5\" --games 120 --workers 6 "
          f"--out outputs/arena/gate_richbc_dagger_v5.csv")


if __name__ == "__main__":
    main()
