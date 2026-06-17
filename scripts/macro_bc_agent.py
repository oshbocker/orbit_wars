"""Slice 3: the macro-action inference agent.

The OrbitNet selector picks, per owned source, WHICH target to attack (or hold). Those
picks are injected into v5's planner via the ``_SELECTOR_FN`` hook so that v5's EXACT
candidate generation, ``intercept_angle`` and ``safe_drain`` sizing execute them — the net
never emits a continuous parameter. This is the "learn the selection, compute the
parameters exactly" architecture (slices 1-2 train the selector; slice 4 gates it).
"""
from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.game_types import parse_observation  # noqa: E402
from v2.config import load_v2_config  # noqa: E402
from v2.features import encode_features  # noqa: E402
from v2.model import OrbitNet  # noqa: E402

_counter = itertools.count()


def _load_v5_module():
    """Exec a FRESH copy of agents/v5/main.py (it keeps module-level runtime state)."""
    main_py = ROOT / "agents" / "v5" / "main.py"
    modname = f"_v5macro_{next(_counter)}"
    spec = importlib.util.spec_from_file_location(modname, main_py)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


def _comet_args(obs):
    cids = obs.get("comet_planet_ids") if isinstance(obs, dict) else getattr(obs, "comet_planet_ids", None)
    cdata = obs.get("comets") if isinstance(obs, dict) else getattr(obs, "comets", None)
    cids = [int(x) for x in cids] if cids is not None else None
    return cids, cdata


def build_macro_agent(ckpt_path: str, config_path: str | None = None):
    """Return an agent(obs, config) that selects targets with OrbitNet and executes
    them through v5's exact planner (via the _SELECTOR_FN hook)."""
    device = torch.device("cpu")
    ck = torch.load(ckpt_path, map_location=device)
    cfg = load_v2_config(config_path or ck.get("config", "configs/v2_exit.yaml"))
    model = OrbitNet(cfg.model).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    mod = _load_v5_module()
    holder: dict = {"logits": None, "fire": None}

    def selector(obs_tensors, cand_src, cand_tgt_slot, cand_valid, cand_score=None):
        # cand_score (exact per-candidate Δnet) is unused here: the macro path overrides the
        # score wholesale and relies on the legacy ROI-off + per-source fire gate.
        logits = holder["logits"]  # [P, P+1]
        fire = holder["fire"]      # [P] bool
        if logits is None:
            return None
        planet_ids = obs_tensors["planets"][..., 0].long()        # [P_rows] -> planet id
        src_pid = planet_ids[cand_src.squeeze(-1)]                 # [C]
        tgt_pid = planet_ids[cand_tgt_slot]                        # [C]
        Pn = logits.shape[0]
        src_pid = src_pid.clamp(0, Pn - 1)
        tgt_pid = tgt_pid.clamp(0, Pn - 1)
        sc = logits[src_pid, tgt_pid + 1]                          # net's source->target logit
        f = fire[src_pid]
        return torch.where(f, sc, torch.full_like(sc, float("-inf")))

    mod._SELECTOR_FN = selector

    @torch.inference_mode()
    def _net_decision(obs):
        state = parse_observation(obs)
        cids, cdata = _comet_args(obs)
        feats = encode_features(state, cfg.env, comet_ids=cids, comets_data=cdata)
        pf = torch.from_numpy(feats.planet_features).unsqueeze(0)
        gf = torch.from_numpy(feats.global_features).unsqueeze(0)
        pm = torch.from_numpy(feats.planet_mask).unsqueeze(0)
        om = torch.from_numpy(feats.own_mask).unsqueeze(0)
        out = model(pf, gf, pm, om)            # NO reachability mask (matches training)
        logits = out.logits[0]                 # [P, P+1]
        pred = logits.argmax(-1)               # [P]; index 0 = hold
        fire = (pred > 0) & om[0]              # owned source whose best is a target
        return logits, fire

    def agent(obs, config=None):
        holder["logits"], holder["fire"] = _net_decision(obs)
        return mod.agent(obs)

    return agent


if __name__ == "__main__":
    # Smoke: one real 2P game vs producer, report result.
    import argparse

    from kaggle_environments import make

    from agents import load_named_agent

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/tmp/macro_smoke.pt")
    ap.add_argument("--config", default="configs/v2_exit.yaml")
    ap.add_argument("--seed", type=int, default=20000)
    args = ap.parse_args()

    macro = build_macro_agent(args.ckpt, args.config)
    env = make("orbit_wars", configuration={"randomSeed": args.seed})
    env.run([macro, load_named_agent("producer")])
    last = env.steps[-1]
    r0, r1 = last[0]["reward"], last[1]["reward"]
    print(f"macro vs producer (seed {args.seed}): reward {r0} vs {r1}  "
          f"-> {'macro' if r0 > r1 else 'producer' if r1 > r0 else 'tie'}  "
          f"({len(env.steps)} steps)")
