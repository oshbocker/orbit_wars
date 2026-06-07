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

IMPORTANT: `agent()` must remain the LAST callable defined at module level —
Kaggle's loader returns the last callable in the module namespace. The bundle
packages are imported at top (before any `def`), so `agent` stays last.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Ensure the bundled `v2/` and `src/` packages are importable. On Kaggle the
# agent runs as `/kaggle_simulations/agent/main.py`, and that directory is NOT
# reliably on sys.path when our deferred imports run — without this, an import of
# `v2.*`/`src.*` raises `ModuleNotFoundError: No module named 'v2'`.
#
# Kaggle's loader (kaggle_environments/agent.py `get_last_callable`) does:
#     sys.path.append(os.path.dirname(path))   # adds /kaggle_simulations/agent
#     exec(code_object, env)                   # runs THIS module body
#     sys.path.pop()                           # ...then REMOVES it again
# and it execs into a bare `env = {}`, so `__file__` is NOT defined here. Two
# consequences we must defend against:
#   1. The dir is already on sys.path during exec, so a `if _d not in sys.path`
#      guard would no-op — then the trailing pop() orphans us and any import that
#      runs LATER (e.g. lazily inside the first agent() call) fails. So we
#      FORCE-insert at position 0; the pop() removes Kaggle's trailing copy and
#      leaves ours intact.
#   2. To be doubly safe, we import the bundle packages at module top-level
#      (below), i.e. DURING exec while the path is guaranteed present, so they
#      land in sys.modules and never need re-resolution after the pop().
try:
    _AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _AGENT_DIR = "/kaggle_simulations/agent"
for _d in ("/kaggle_simulations/agent", _AGENT_DIR):
    if _d:
        sys.path.insert(0, _d)  # unconditional — survives Kaggle's sys.path.pop()

import torch

from src.game_types import parse_observation as _parse

# Eagerly import the bundled packages NOW (during exec, while the agent dir is on
# sys.path) so they are cached in sys.modules before Kaggle pops the path entry.
from v2.actions import decode_actions as _decode
from v2.config import V2Config, load_v2_config
from v2.features import encode_features as _encode
from v2.model import OrbitNet

# Singleton state (model + config loaded on first call).
_model = None
_cfg = None
_device = None


def _find(paths: list[str]) -> str | None:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _resolve_config():
    """Load the V2Config matching the trained checkpoint, or a sane default."""
    cfg_path = _find(
        [
            os.environ.get("V2_CONFIG", ""),
            "submission_config.yaml",
            os.path.join(_AGENT_DIR, "submission_config.yaml"),
            "/kaggle_simulations/agent/submission_config.yaml",
        ]
    )
    if cfg_path is not None:
        return load_v2_config(cfg_path)
    # Fallback: defaults (v2 architecture, features off). Works for v2 checkpoints.
    return V2Config()


def _init() -> None:
    global _model, _cfg, _device
    _device = torch.device("cpu")

    _cfg = _resolve_config()

    ckpt_path = _find(
        [
            os.environ.get("V2_CHECKPOINT", ""),
            "ckpt_last.pt",
            os.path.join(_AGENT_DIR, "ckpt_last.pt"),
            "/kaggle_simulations/agent/ckpt_last.pt",
        ]
    )
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
    assert _cfg is not None  # _init() populates _model and _cfg together

    state = _parse(obs)
    features = _encode(state, _cfg.env, comet_ids=_comet_ids(obs), comets_data=_comets_data(obs))

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
