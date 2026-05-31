"""V3 Kaggle submission agent — uses the REAL feature pipeline (no inlining).

Unlike v2/agent.py (which inlines a minimal feature encoder), this agent imports
the actual training-time code (`encode_features`, `OrbitNet`, `decode_actions`),
so the submitted policy sees byte-identical features to what it trained on —
including the v3 Tier-1 enhancements (pairwise travel-time/required-ships features
and comet targeting). This eliminates train/deploy feature drift.

Requirements (bundled by the Colab submission cell):
  - the `v2/` and `src/` packages on the path
  - the trained checkpoint `ckpt_last.pt`
  - a config YAML describing the feature flags / model dims used in training
    (so we build the matching OrbitNet and feature set)

Resolution order for each artifact (first hit wins):
  checkpoint : $V2_CHECKPOINT, ./ckpt_last.pt, /kaggle_simulations/agent/ckpt_last.pt
  config     : $V2_CONFIG,    ./submission_config.yaml, /kaggle_simulations/agent/submission_config.yaml

IMPORTANT: `agent()` must remain the LAST callable defined at module level.
"""
from __future__ import annotations

import os
from typing import Any

import torch

# Singleton state (loaded on first call).
_model = None
_cfg = None
_device = None
_encode = None
_decode = None
_parse = None


def _find(paths: list[str]) -> str | None:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _resolve_config():
    """Load the V2Config matching the trained checkpoint, or a sane default."""
    from v2.config import V2Config, load_v2_config

    cfg_path = _find([
        os.environ.get("V2_CONFIG", ""),
        "submission_config.yaml",
        "/kaggle_simulations/agent/submission_config.yaml",
    ])
    if cfg_path is not None:
        return load_v2_config(cfg_path)
    # Fallback: defaults (v2 architecture, features off). Works for v2 checkpoints.
    return V2Config()


def _init() -> None:
    global _model, _cfg, _device, _encode, _decode, _parse
    _device = torch.device("cpu")

    from v2.actions import decode_actions
    from v2.features import encode_features
    from v2.model import OrbitNet
    from src.game_types import parse_observation

    _encode, _decode, _parse = encode_features, decode_actions, parse_observation
    _cfg = _resolve_config()

    ckpt_path = _find([
        os.environ.get("V2_CHECKPOINT", ""),
        "ckpt_last.pt",
        "/kaggle_simulations/agent/ckpt_last.pt",
    ])
    if ckpt_path is None:
        raise FileNotFoundError("No checkpoint found (ckpt_last.pt).")

    _model = OrbitNet(_cfg.model).to(_device)
    ckpt = torch.load(ckpt_path, map_location=_device, weights_only=True)
    _model.load_state_dict(ckpt["model"])
    _model.eval()


def _comets_data(obs: Any):
    if hasattr(obs, "comets"):
        return getattr(obs, "comets", None)
    if isinstance(obs, dict):
        return obs.get("comets")
    return None


def _comet_ids(obs: Any):
    ids = getattr(obs, "comet_planet_ids", None)
    if ids is None and isinstance(obs, dict):
        ids = obs.get("comet_planet_ids")
    return [int(x) for x in ids] if ids is not None else None


def agent(obs, config=None):
    """V3 Kaggle agent. MUST be the last callable in the file."""
    if _model is None:
        _init()

    state = _parse(obs)
    features = _encode(state, _cfg.env, comet_ids=_comet_ids(obs),
                       comets_data=_comets_data(obs))

    with torch.inference_mode():
        pf = torch.from_numpy(features.planet_features).unsqueeze(0).to(_device)
        gf = torch.from_numpy(features.global_features).unsqueeze(0).to(_device)
        pm = torch.from_numpy(features.planet_mask).unsqueeze(0).to(_device)
        om = torch.from_numpy(features.own_mask).unsqueeze(0).to(_device)
        rm = torch.from_numpy(features.reachability_mask).unsqueeze(0).to(_device)
        pairf = None
        if features.pair_features is not None:
            pairf = torch.from_numpy(features.pair_features).unsqueeze(0).to(_device)
        output = _model(pf, gf, pm, om, rm, pairf)

    return _decode(output, features, state, _cfg.env, deterministic=True)
