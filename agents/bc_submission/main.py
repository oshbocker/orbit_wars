"""BC (win-weighted gate-head OrbitNet) Kaggle submission agent.

Self-contained bundle entry point. Reproduces the EXACT decode/executor of
`agents/external/bc_teacher.py` (gate -> pointer -> capture-sizing), so the
submitted agent behaves byte-identically to what we gate locally with
`scripts/winbc_probe.py` / `scripts/arena.py`. It is NOT the sampling
`decode_actions` path used by `v2/agent_v3.py`.

Bundle layout (archive root, assembled by `scripts/build_bc_bundle.py`):
    main.py                 <- this file (agent() is the LAST callable)
    contested_drainer.py    <- standalone obs-parse / aim helpers
    bc_config.py            <- DRAIN / GATE_THR chosen by the local sweep
    ckpt.pt                 <- trained winbc gate-head checkpoint (carries `arch`)
    submission_config.yaml  <- v2 config (env feature flags)
    v2/  src/               <- full packages (feature pipeline + model)

The architecture is reconstructed from the checkpoint (saved `arch`, else inferred
from weight shapes) — the training-time scale overrides are NOT in the config yaml,
so trusting it would rebuild a wrong-sized net and silently load a random one.

IMPORTANT: `agent()` must remain the LAST callable defined at module level —
Kaggle's loader returns the last callable in the module namespace.
"""

from __future__ import annotations

import math
import os
import sys

# Make the bundled packages importable, surviving Kaggle's loader sys.path.pop()
# (same defense as v2/agent_v3.py: force-insert at 0, import packages during exec).
try:
    _AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _AGENT_DIR = "/kaggle_simulations/agent"
for _d in ("/kaggle_simulations/agent", _AGENT_DIR):
    if _d:
        sys.path.insert(0, _d)

import contested_drainer as CD
import torch

from src.game_types import parse_observation
from v2.config import load_v2_config
from v2.features import encode_features
from v2.model import OrbitNet

# --- Executor knobs (chosen by the local drain/threshold sweep) -------------------
# Defaults are overridden by bundled bc_config.py (written at build time), then by
# env vars (for local A/B). Sizing constants mirror bc_teacher.py exactly.
_DRAIN = "min"
_GATE_THR = 0.5
try:
    import bc_config as _BC  # bundled

    _DRAIN = getattr(_BC, "DRAIN", _DRAIN)
    _GATE_THR = float(getattr(_BC, "GATE_THR", _GATE_THR))
except Exception:
    pass
_DRAIN = os.environ.get("OW_BC_DRAIN", _DRAIN).lower()
_GATE_THR = float(os.environ.get("OW_BC_GATE_THR", _GATE_THR))

_CAPTURE_MARGIN = 2.0
_REINFORCE_KEEP = 1.0
_MIN_FLEET = 3

_MODEL = None
_CFG = None
_GATE = False


def _find(paths: list[str]) -> str | None:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _load() -> bool:
    global _MODEL, _CFG, _GATE
    if _MODEL is not None:
        return True
    ckpt_path = _find(
        [
            os.environ.get("OW_BC_TEACHER_CKPT", ""),
            "ckpt.pt",
            os.path.join(_AGENT_DIR, "ckpt.pt"),
            "/kaggle_simulations/agent/ckpt.pt",
        ]
    )
    cfg_path = _find(
        [
            "submission_config.yaml",
            os.path.join(_AGENT_DIR, "submission_config.yaml"),
            "/kaggle_simulations/agent/submission_config.yaml",
        ]
    )
    if ckpt_path is None or cfg_path is None:
        return False

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = load_v2_config(cfg_path)
    _GATE = bool(ck.get("gate_head", False))
    if _GATE:
        cfg.model.launch_gate_head = True

    # Reconstruct the true architecture (overrides aren't in the config yaml):
    # prefer the saved `arch`, else infer embed/layers/ff from weight shapes.
    sd = ck["model"]
    arch = ck.get("arch", {})
    cfg.model.embed_dim = int(arch.get("embed_dim", sd["planet_embed.0.weight"].shape[0]))
    cfg.model.n_layers = int(
        arch.get("n_layers", 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("transformer_blocks.")))
    )
    cfg.model.ff_dim = int(arch.get("ff_dim", sd["transformer_blocks.0.ff.0.weight"].shape[0]))
    nh = arch.get("n_heads") or os.environ.get("OW_BC_NHEADS")
    if nh is not None:
        cfg.model.n_heads = int(nh)
    elif cfg.model.embed_dim % cfg.model.n_heads != 0:
        cfg.model.n_heads = next(h for h in (8, 4, 2, 1) if cfg.model.embed_dim % h == 0)

    model = OrbitNet(cfg.model)
    res = model.load_state_dict(sd, strict=False)
    missing = [k for k in res.missing_keys if "gate_head" not in k and "frac" not in k]
    if res.unexpected_keys or missing:
        raise RuntimeError(f"BC ckpt load mismatch: missing={missing[:6]} unexpected={list(res.unexpected_keys)[:6]}")
    model.eval()
    _MODEL, _CFG = model, cfg
    return True


def _comet_args(obs):
    cids = obs.get("comet_planet_ids") if isinstance(obs, dict) else getattr(obs, "comet_planet_ids", None)
    cdata = obs.get("comets") if isinstance(obs, dict) else getattr(obs, "comets", None)
    cids = [int(x) for x in cids] if cids is not None else None
    return cids, cdata


def _net_pred(obs):
    """Per planet-id row: 0 = hold, else target_id = idx-1. Gate fires -> pointer argmax."""
    state = parse_observation(obs)
    cids, cdata = _comet_args(obs)
    feats = encode_features(state, _CFG.env, comet_ids=cids, comets_data=cdata)
    pf = torch.from_numpy(feats.planet_features).unsqueeze(0)
    gf = torch.from_numpy(feats.global_features).unsqueeze(0)
    pm = torch.from_numpy(feats.planet_mask).unsqueeze(0)
    om = torch.from_numpy(feats.own_mask).unsqueeze(0)
    with torch.inference_mode():
        out = _MODEL(pf, gf, pm, om)
    logits = out.logits[0]
    tgt = logits[:, 1:]
    best_t = tgt.argmax(-1)
    if _GATE and out.gate_logits is not None:
        fire = torch.sigmoid(out.gate_logits[0]) > _GATE_THR
    else:
        hold = logits[:, :1]
        best_v = tgt.gather(1, best_t[:, None]).squeeze(1)
        fire = (best_v - hold.squeeze(1)) >= 0.0
    pred = torch.where(fire, best_t + 1, torch.zeros_like(best_t))
    return pred.tolist()


def agent(obs, config=None) -> list:
    """BC Kaggle agent. MUST be the last callable in the file."""
    if not _load():
        return CD.plan(obs, CD.CONFIG)

    pid = int(CD._obs_get(obs, "player", 0))
    ang_vel = float(CD._obs_get(obs, "angular_velocity", 0.0) or 0.0)
    planets = CD._planets(obs)
    if not planets:
        return []
    by_id = {p["id"]: p for p in planets}
    owned = sorted((p for p in planets if p["owner"] == pid), key=lambda p: -p["ships"])
    if not owned:
        return []

    pred = _net_pred(obs)
    moves: list = []
    for src in owned:
        sid = src["id"]
        if sid >= len(pred):
            continue
        k = pred[sid]
        if k <= 0:
            continue
        tgt = by_id.get(k - 1)
        if tgt is None or tgt["id"] == sid:
            continue
        if tgt["owner"] == pid:
            ships_w = int(src["ships"] - _REINFORCE_KEEP)
            if ships_w < _MIN_FLEET:
                continue
            aimed = CD._aim(src, tgt, ang_vel, est_ships=ships_w)
            if aimed is None:
                continue
            _, angle, _, _ = aimed
            moves.append([sid, float(angle), ships_w])
            continue
        aimed = CD._aim(src, tgt, ang_vel, est_ships=max(tgt["ships"], 8.0))
        if aimed is None:
            continue
        eta, angle, _, _ = aimed
        if tgt["owner"] == -1:
            need = tgt["ships"] + _CAPTURE_MARGIN
        else:
            need = tgt["ships"] + tgt["prod"] * eta + _CAPTURE_MARGIN
        need = int(math.ceil(need))
        if need > src["ships"]:
            continue
        ships_w = max(_MIN_FLEET, need)
        if _DRAIN == "full":
            ships_w = max(ships_w, int(src["ships"] - _REINFORCE_KEEP))
        if ships_w > src["ships"]:
            continue
        moves.append([sid, float(angle), int(ships_w)])
    return moves
