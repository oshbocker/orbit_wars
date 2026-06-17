"""Slice 3 (rich): inference agent for the rich-representation selector.

In ONE v5 forward per turn: plan_lite_waves fills _FEATURE_SINK with producer's exact
projection grid, then _SELECTOR_FN runs the RichSelector (v2 features + that grid) and
overrides the candidate score, so v5 executes the net's picks with exact intercept/safe_drain.
Rolling cache (matches training capture); v5 byte-identical when the hooks are unset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.producer_features import ProducerFeatureExtractor, load_v5_module  # noqa: E402
from scripts.rich_bc_train import RichSelector, _comet_args, _pair_features  # noqa: E402
from src.game_types import parse_observation  # noqa: E402
from v2.config import load_v2_config  # noqa: E402
from v2.features import encode_features  # noqa: E402


def build_rich_agent(ckpt_path: str, config_path: str | None = None):
    dev = torch.device("cpu")
    ck = torch.load(ckpt_path, map_location=dev)
    cfg = load_v2_config(config_path or ck.get("config", "configs/v2_exit.yaml"))
    cfg.model.use_pair_features = True
    cfg.model.pair_feat_dim = int(ck.get("pair_feat_dim", 4))
    model = RichSelector(cfg.model).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    mod = load_v5_module()
    ext = ProducerFeatureExtractor(max_planets=cfg.env.max_planets)
    P = cfg.env.max_planets
    stash: dict = {}

    @torch.inference_mode()
    def selector(obs_tensors, cand_src, cand_tgt_slot, cand_valid, cand_score):
        # Rich residual: return producer's EXACT per-candidate Δnet (`cand_score`) plus the
        # net's learned delta (0 at init => byte-identical to v5), on the real score scale.
        # v5's real ROI gate (kept on via _SELECTOR_KEEP_ROI) then decides firing/order, so
        # firing discipline matches v5 instead of the old z-scored "fire all" path.
        sink = mod._FEATURE_SINK
        v2 = stash.get("v2")
        if not sink or v2 is None:
            return None
        grid = ext._densify(sink)
        pair = torch.from_numpy(_pair_features(grid, P)).unsqueeze(0)   # [1,P,P,F] net input
        delta = model(v2[0], v2[1], v2[2], v2[3], pair)[0]            # [P,P+1]; 0 at init
        planet_ids = obs_tensors["planets"][..., 0].long()
        n = delta.shape[0]
        src_pid = planet_ids[cand_src.squeeze(-1)].clamp(0, n - 1)
        tgt_pid = planet_ids[cand_tgt_slot].clamp(0, n - 1)
        d = delta[src_pid, tgt_pid + 1]                                # [C] per-candidate delta
        return cand_score + d                                         # real-scale; invalid masked by hook

    mod._SELECTOR_FN = selector
    mod._SELECTOR_KEEP_ROI = True

    @torch.inference_mode()
    def _v2feats(obs):
        state = parse_observation(obs)
        cids, cdata = _comet_args(obs)
        feats = encode_features(state, cfg.env, comet_ids=cids, comets_data=cdata)
        return (
            torch.from_numpy(feats.planet_features).unsqueeze(0),
            torch.from_numpy(feats.global_features).unsqueeze(0),
            torch.from_numpy(feats.planet_mask).unsqueeze(0),
            torch.from_numpy(feats.own_mask).unsqueeze(0),
        )

    def agent(obs, config=None):
        stash["v2"] = _v2feats(obs)
        mod._FEATURE_SINK = {}                # enable grid capture for this turn
        try:
            return mod.agent(obs)
        finally:
            mod._FEATURE_SINK = None

    return agent


if __name__ == "__main__":
    import argparse

    from kaggle_environments import make

    from agents import load_named_agent

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/tmp/rich_live.pt")
    ap.add_argument("--config", default="configs/v2_exit.yaml")
    ap.add_argument("--opponent", default="v5")
    ap.add_argument("--seed", type=int, default=20000)
    args = ap.parse_args()

    rich = build_rich_agent(args.ckpt, args.config)
    env = make("orbit_wars", configuration={"randomSeed": args.seed})
    env.run([rich, load_named_agent(args.opponent)])
    last = env.steps[-1]
    r0, r1 = last[0]["reward"], last[1]["reward"]
    print(f"rich vs {args.opponent} (seed {args.seed}): {r0} vs {r1} -> "
          f"{'rich' if r0 > r1 else args.opponent if r1 > r0 else 'tie'} ({len(env.steps)} steps)")
