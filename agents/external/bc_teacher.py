"""BC-teacher FIXTURE: a standalone clone of a strong NON-producer ladder agent.

The contested off-mirror instrument. A pointer net (v2 OrbitNet) is behavior-cloned from a
real top-tier non-producer teacher's replays (target SELECTION only — the AlphaStar pointer
decomposition; see ``scripts/teacher_bc.py``). At inference the net picks, per owned planet,
WHICH target to attack (or hold); the angle + ship count are computed ANALYTICALLY here
(intercept aim + capture-minimal sizing), so the clone never regresses a continuous
parameter (the failure that capped prior BC at 3%) AND keeps the teacher's partial-send
(half-drain) signature — distinct from v5's full-drain.

CRUCIALLY this executor is STANDALONE — it does NOT route through v5's planner (that would
size with v5's full-drain ``safe_drain`` and collapse the clone back toward the producer
mirror). It reuses only ``contested_drainer``'s geometry (sun avoidance, orbit-lead
intercept, capture-floor sizing).

NOT a competitor — a deterministic MEASUREMENT FIXTURE for the off-mirror gate. Checkpoint
path overridable via ``OW_BC_TEACHER_CKPT``; absent checkpoint => falls back to the
heuristic ``contested_drainer`` so the fixture still loads.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

from agents.external import contested_drainer as CD

_REPO = Path(__file__).resolve().parent.parent.parent
_CKPT = os.environ.get("OW_BC_TEACHER_CKPT", str(_REPO / "outputs" / "checkpoints" / "bc_teacher" / "ckpt.pt"))

# Aggression knob (combats BC's all-hold bias): fire at a source's best NON-hold target
# when its logit is within ``_FIRE_MARGIN`` of the hold logit, rather than strict argmax
# (margin 0 == argmax). The net ranks targets well but is weak on the binary hold/launch
# decision, so a positive margin recovers the teacher's activity. Tune via OW_BC_FIRE_MARGIN.
_FIRE_MARGIN = float(os.environ.get("OW_BC_FIRE_MARGIN", "0.0"))

# Separate-gate decode threshold: when the checkpoint has a launch/no-launch gate head
# (winbc --gate_head), a source fires iff sigmoid(gate_logit) > _GATE_THR. This replaces the
# _FIRE_MARGIN band-aid (which only existed because hold-as-a-class made the launch decision
# weak); with a real gate, the act decision is a first-class output. Tune via OW_BC_GATE_THR.
_GATE_THR = float(os.environ.get("OW_BC_GATE_THR", "0.5"))

# Sizing: enemy/neutral captures are capture-minimal (the half-drain signature); a pick of
# an OWNED target is a reinforce/consolidation send (forward most of the garrison).
_CAPTURE_MARGIN = 2.0
_REINFORCE_KEEP = 1.0
_MIN_FLEET = 3

# Executor-sizing ablation (confound control for the winbc regime-check): the capture-minimal
# default is a half-drain prior — a documented partial-send tempo tax (contested-instrument
# memory). OW_BC_DRAIN=full forwards near the full garrison on AFFORDABLE captures instead, so
# we can tell whether the clone's losses come from the NET's selections or from the executor's
# under-sizing. "min" reproduces the prior half-drain behaviour exactly.
_DRAIN = os.environ.get("OW_BC_DRAIN", "min").lower()

_MODEL = None
_CFG = None
_TORCH = None
_GATE = False  # checkpoint has a separate launch/no-launch gate head


def _load():
    global _MODEL, _CFG, _TORCH, _GATE
    if _MODEL is not None:
        return True
    if not os.path.exists(_CKPT):
        return False
    import torch

    from src.game_types import parse_observation  # noqa: F401  (imported lazily elsewhere)
    from v2.config import load_v2_config
    from v2.model import OrbitNet

    ck = torch.load(_CKPT, map_location="cpu")
    cfg = load_v2_config(ck.get("config", "configs/v2_exit.yaml"))
    _GATE = bool(ck.get("gate_head", False))
    if _GATE:
        cfg.model.launch_gate_head = True

    # The training-time scale overrides (--embed-dim/--n-layers/--ff-dim/--n-heads) are NOT
    # captured by the checkpoint's `config` string, which just names a yaml at its DEFAULT dims.
    # Trusting it silently rebuilds a wrong-sized OrbitNet whose weights all mismatch -> with a
    # lenient load the deployed net is RANDOMLY INITIALISED (it never launches). Reconstruct the
    # true architecture: prefer explicit saved `arch`, else infer from the weight shapes.
    sd = ck["model"]
    arch = ck.get("arch", {})
    cfg.model.embed_dim = int(arch.get("embed_dim", sd["planet_embed.0.weight"].shape[0]))
    cfg.model.n_layers = int(arch.get("n_layers", 1 + max(
        int(k.split(".")[1]) for k in sd if k.startswith("transformer_blocks."))))
    cfg.model.ff_dim = int(arch.get("ff_dim", sd["transformer_blocks.0.ff.0.weight"].shape[0]))
    # n_heads doesn't change any weight shape, so it can't be inferred — take the saved value,
    # else env override, else the yaml default only if it divides embed_dim, else a safe divisor.
    nh = arch.get("n_heads") or os.environ.get("OW_BC_NHEADS")
    if nh is not None:
        cfg.model.n_heads = int(nh)
    elif cfg.model.embed_dim % cfg.model.n_heads != 0:
        cfg.model.n_heads = next(h for h in (8, 4, 2, 1) if cfg.model.embed_dim % h == 0)

    model = OrbitNet(cfg.model)
    # strict on the core net (FAIL LOUD on any future arch drift); the optional gate/frac heads
    # are gated by flags so a missing one is the only tolerated discrepancy.
    res = model.load_state_dict(sd, strict=False)
    unexpected = list(res.unexpected_keys)
    missing = [k for k in res.missing_keys if "gate_head" not in k and "frac" not in k]
    if unexpected or missing:
        raise RuntimeError(
            f"bc_teacher checkpoint load mismatch (arch reconstruction failed): "
            f"missing={missing[:6]} unexpected={unexpected[:6]}")
    model.eval()
    _MODEL, _CFG, _TORCH = model, cfg, torch
    return True


def _comet_args(obs):
    cids = obs.get("comet_planet_ids") if isinstance(obs, dict) else getattr(obs, "comet_planet_ids", None)
    cdata = obs.get("comets") if isinstance(obs, dict) else getattr(obs, "comets", None)
    cids = [int(x) for x in cids] if cids is not None else None
    return cids, cdata


def _net_pred(obs):
    """Per planet-id (== feature row), the net's pick: 0 = hold, else target_id = idx-1.

    Fires at the best NON-hold target when (best_target_logit - hold_logit) >= -_FIRE_MARGIN
    (margin 0 reproduces argmax). The half-drainer launches ~1.1 waves/turn, so a modest
    positive margin recovers activity without over-launching.
    """
    from src.game_types import parse_observation
    from v2.features import encode_features

    torch = _TORCH
    state = parse_observation(obs)
    cids, cdata = _comet_args(obs)
    feats = encode_features(state, _CFG.env, comet_ids=cids, comets_data=cdata)
    pf = torch.from_numpy(feats.planet_features).unsqueeze(0)
    gf = torch.from_numpy(feats.global_features).unsqueeze(0)
    pm = torch.from_numpy(feats.planet_mask).unsqueeze(0)
    om = torch.from_numpy(feats.own_mask).unsqueeze(0)
    with torch.inference_mode():
        out = _MODEL(pf, gf, pm, om)            # NO reachability mask (matches training)
    logits = out.logits[0]                       # [P, P+1]; col 0 = hold, col k>0 => target row k-1
    tgt = logits[:, 1:]                           # [P,P]
    best_t = tgt.argmax(-1)                        # [P] best target row
    if _GATE and out.gate_logits is not None:
        # Decoupled gate: fire iff P(launch) > threshold, then take the pointer argmax.
        fire = torch.sigmoid(out.gate_logits[0]) > _GATE_THR
    else:
        # hold-as-a-class fallback: fire when best target beats hold within _FIRE_MARGIN.
        hold = logits[:, :1]
        best_v = tgt.gather(1, best_t[:, None]).squeeze(1)
        fire = (best_v - hold.squeeze(1)) >= -_FIRE_MARGIN
    pred = torch.where(fire, best_t + 1, torch.zeros_like(best_t))   # 0 = hold else target+1
    return pred.tolist()


def agent(obs, config=None) -> list:
    if not _load():
        # No trained checkpoint yet: behave as the heuristic strong drainer so the
        # fixture still loads and plays (used only before/without training).
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
            continue  # net says hold
        tgt = by_id.get(k - 1)
        if tgt is None or tgt["id"] == sid:
            continue
        if tgt["owner"] == pid:
            # reinforce / forward logistics: send most of the garrison.
            ships_w = int(src["ships"] - _REINFORCE_KEEP)
            if ships_w < _MIN_FLEET:
                continue
            aimed = CD._aim(src, tgt, ang_vel, est_ships=ships_w)
            if aimed is None:
                continue
            _, angle, _, _ = aimed
            moves.append([sid, float(angle), ships_w])
            continue
        # enemy/neutral capture: capture-minimal sizing (the half-drain signature).
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
            continue  # can't take it from here this turn
        ships_w = max(_MIN_FLEET, need)
        if _DRAIN == "full":
            # Forward near the full garrison once the capture is affordable (full-drain
            # tempo) rather than the capture-minimal half-drain. Affordability is already
            # guaranteed by need <= src["ships"] above.
            ships_w = max(ships_w, int(src["ships"] - _REINFORCE_KEEP))
        if ships_w > src["ships"]:
            continue
        moves.append([sid, float(angle), int(ships_w)])
    return moves
