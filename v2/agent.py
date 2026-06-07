"""V2 Kaggle submission agent.

Self-contained: loads checkpoint on first call.
IMPORTANT: agent() must be the LAST callable in this file.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

# Ensure the bundled `v2/` package is importable. On Kaggle the agent runs as
# `/kaggle_simulations/agent/main.py`, and that directory is NOT automatically
# on sys.path — without this, `from v2.model import ...` raises
# `ModuleNotFoundError: No module named 'v2'`.
# Kaggle runs the agent via `exec(code, env)`, so `__file__` is NOT defined in
# that namespace — fall back to the known agent directory.
try:
    _AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _AGENT_DIR = "/kaggle_simulations/agent"
for _d in (_AGENT_DIR, "/kaggle_simulations/agent"):
    if _d and _d not in sys.path:
        sys.path.insert(0, _d)

# ── Inline constants (avoid importing src at submission time) ──────────────
_BOARD_SIZE = 100.0
_SUN_X, _SUN_Y = 50.0, 50.0
_SUN_RADIUS = 10.0
_SUN_SAFE_RADIUS = _SUN_RADIUS + 2.0
_MAX_SHIP_SPEED = 6.0
_MAX_PLANETS = 40
_PLANET_FEAT_DIM = 22
_GLOBAL_FEAT_DIM = 8
_ALLOCATION_THRESHOLD = 0.05
_MIN_SHIPS = 1
_SHIP_FRACTIONS = [0.25, 0.5, 0.75, 1.0]  # must match env.ship_fractions used in training

# ── Singleton state ────────────────────────────────────────────────────────
_model = None
_device = None
_cfg = None


def _fleet_speed(ships: float) -> float:
    if ships <= 1:
        return 1.0
    return 1.0 + (_MAX_SHIP_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def _passes_through_sun(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - _SUN_X, y1 - _SUN_Y
    a = dx * dx + dy * dy
    if a == 0:
        return False
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - _SUN_SAFE_RADIUS**2
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 < t2)


def _can_reach(sx, sy, tx, ty):
    """Check if a fleet from (sx,sy) can reach (tx,ty) without hitting the sun."""
    return not _passes_through_sun(sx, sy, tx, ty)


def _is_valid_target(src, tgt, player, step):
    """Check if sending a fleet from src to tgt is a valid action.

    Combines: sun avoidance, takeover viability, arrival time.
    Own planets (reinforcement) bypass viability.
    """
    if not _can_reach(src["x"], src["y"], tgt["x"], tgt["y"]):
        return False

    # Arrival time check
    dist = math.hypot(src["x"] - tgt["x"], src["y"] - tgt["y"])
    speed = _fleet_speed(src["ships"])
    eta = dist / speed if speed > 0 else 999.0
    steps_remaining = max(0, 498 - step)
    if eta > steps_remaining:
        return False

    # Own planets: reinforcement always valid
    if tgt["owner"] == player:
        return True

    # Takeover viability for enemy/neutral
    prod_growth = tgt["production"] * math.ceil(eta) if tgt["owner"] >= 0 else 0.0
    return src["ships"] >= tgt["ships"] + prod_growth + 1


def _parse_planet(p):
    if hasattr(p, "production"):
        return {
            "id": int(p.id),
            "owner": int(p.owner),
            "x": float(p.x),
            "y": float(p.y),
            "radius": float(p.radius),
            "ships": int(p.ships),
            "production": int(p.production),
        }
    if isinstance(p, dict):
        return {
            "id": int(p["id"]),
            "owner": int(p["owner"]),
            "x": float(p["x"]),
            "y": float(p["y"]),
            "radius": float(p["radius"]),
            "ships": int(p["ships"]),
            "production": int(p["production"]),
        }
    return {
        "id": int(p[0]),
        "owner": int(p[1]),
        "x": float(p[2]),
        "y": float(p[3]),
        "radius": float(p[4]),
        "ships": int(p[5]),
        "production": int(p[6]),
    }


def _obs_get(obs, key, default=None):
    if hasattr(obs, key):
        return getattr(obs, key)
    if isinstance(obs, dict):
        return obs.get(key, default)
    return default


def _encode_features(obs):
    player = int(_obs_get(obs, "player", 0))
    step = int(_obs_get(obs, "step", 0))
    angular_velocity = float(_obs_get(obs, "angular_velocity", 0.0))
    raw_planets = _obs_get(obs, "planets", []) or []
    planets = [_parse_planet(p) for p in raw_planets]

    # Build enemy ID list
    enemy_ids = []
    for p in planets:
        if p["owner"] >= 0 and p["owner"] != player and p["owner"] not in enemy_ids:
            enemy_ids.append(p["owner"])

    pf = np.zeros((_MAX_PLANETS, _PLANET_FEAT_DIM), dtype=np.float32)
    pm = np.zeros(_MAX_PLANETS, dtype=bool)
    om = np.zeros(_MAX_PLANETS, dtype=bool)
    planet_map = {}

    for p in planets:
        slot = p["id"]
        if slot < 0 or slot >= _MAX_PLANETS:
            continue
        planet_map[slot] = p
        pm[slot] = True
        om[slot] = p["owner"] == player

        own = [0.0, 0.0, 0.0, 0.0]
        if p["owner"] == player:
            own[0] = 1.0
        elif p["owner"] >= 0:
            idx = enemy_ids.index(p["owner"]) if p["owner"] in enemy_ids else 0
            own[min(idx + 1, 3)] = 1.0

        dx = p["x"] - _SUN_X
        dy = p["y"] - _SUN_Y
        dist_center = math.hypot(dx, dy)
        theta = math.atan2(dy, dx) if dist_center > 0.1 else 0.0

        pf[slot] = [
            1.0,
            0.0,  # exists, orbiting (simplified)
            own[0],
            own[1],
            own[2],
            own[3],
            math.log1p(p["ships"]) / 7.0,
            p["x"] / _BOARD_SIZE,
            p["y"] / _BOARD_SIZE,
            dist_center / 70.7,
            math.sin(theta),
            math.cos(theta),
            p["production"] / 5.0,
            p["radius"] / 4.0,
            0.0,
            0.0,
            0.0,
            0.0,  # incoming fleets (skip for speed)
            0.0,
            0.0,
            0.0,
            0.0,  # incoming ETAs
        ]

    # Global features
    my_ships = my_prod = 0.0
    enemy_ships_total = {}
    enemy_prod_total = {}
    total_planets = 0
    my_planets_count = 0
    for p in planets:
        total_planets += 1
        if p["owner"] == player:
            my_ships += p["ships"]
            my_prod += p["production"]
            my_planets_count += 1
        elif p["owner"] >= 0:
            enemy_ships_total[p["owner"]] = enemy_ships_total.get(p["owner"], 0) + p["ships"]
            enemy_prod_total[p["owner"]] = enemy_prod_total.get(p["owner"], 0) + p["production"]

    raw_fleets = _obs_get(obs, "fleets", []) or []
    for f in raw_fleets:
        fowner = int(getattr(f, "owner", f["owner"] if isinstance(f, dict) else f[1]))
        fships = int(getattr(f, "ships", f["ships"] if isinstance(f, dict) else f[6]))
        if fowner == player:
            my_ships += fships
        elif fowner >= 0:
            enemy_ships_total[fowner] = enemy_ships_total.get(fowner, 0) + fships

    best_enemy_ships = max(enemy_ships_total.values()) if enemy_ships_total else 0.0
    best_enemy_prod = max(enemy_prod_total.values()) if enemy_prod_total else 0.0

    gf = np.array(
        [
            step / 500.0,
            angular_velocity / 0.05,
            math.log1p(my_ships) / 10.0,
            math.log1p(best_enemy_ships) / 10.0,
            my_prod / max(my_prod + best_enemy_prod, 1.0),
            best_enemy_prod / max(my_prod + best_enemy_prod, 1.0),
            my_planets_count / _MAX_PLANETS,
            total_planets / _MAX_PLANETS,
        ],
        dtype=np.float32,
    )

    return pf, gf, pm, om, planet_map


def _init_model():
    global _model, _device, _cfg

    _device = torch.device("cpu")

    # Import model class
    from v2.config import V2ModelConfig
    from v2.model import OrbitNet

    _cfg = V2ModelConfig()

    # Find checkpoint
    ckpt_path = os.environ.get("V2_CHECKPOINT", "ckpt_last.pt")
    if not os.path.exists(ckpt_path):
        # Try common locations
        for candidate in [
            os.path.join(_AGENT_DIR, "ckpt_last.pt"),
            "/kaggle_simulations/agent/ckpt_last.pt",
            "outputs/checkpoints/v2_default/ckpt_last.pt",
        ]:
            if os.path.exists(candidate):
                ckpt_path = candidate
                break

    _model = OrbitNet(_cfg).to(_device)
    ckpt = torch.load(ckpt_path, map_location=_device, weights_only=True)
    _model.load_state_dict(ckpt["model"])
    _model.eval()


def agent(obs, config=None):
    """V2 Kaggle agent. MUST be the last callable in the file."""
    global _model, _device

    if _model is None:
        _init_model()

    pf, gf, pm, om, planet_map = _encode_features(obs)

    with torch.inference_mode():
        pf_t = torch.from_numpy(pf).unsqueeze(0).to(_device)
        gf_t = torch.from_numpy(gf).unsqueeze(0).to(_device)
        pm_t = torch.from_numpy(pm).unsqueeze(0).to(_device)
        om_t = torch.from_numpy(om).unsqueeze(0).to(_device)
        output = _model(pf_t, gf_t, pm_t, om_t)

    logits = output.logits[0]  # [P, P+1]
    frac_logits = output.frac_logits[0]  # [P, P, K]
    player = int(_obs_get(obs, "player", 0))
    step = int(_obs_get(obs, "step", 0))
    moves = []

    for i in range(_MAX_PLANETS):
        if not om[i]:
            continue
        src = planet_map.get(i)
        if src is None or src["ships"] <= 0:
            continue

        # Mask invalid targets (sun / viability / arrival), then argmax over
        # [hold, valid targets]. Factored fraction head sets the fleet size.
        row = logits[i].clone()  # [P+1]
        for j in range(_MAX_PLANETS):
            tgt = planet_map.get(j)
            if (
                tgt is None
                or tgt["id"] == src["id"]
                or not _is_valid_target(src, tgt, player, step)
            ):
                row[j + 1] = float("-inf")
        if not torch.isfinite(row[1:]).any():
            continue

        action = int(row.argmax().item())
        if action == 0:
            continue  # model chose hold
        j = action - 1
        tgt = planet_map.get(j)
        if tgt is None:
            continue

        fbin = int(frac_logits[i, j].argmax().item())
        fbin = max(0, min(len(_SHIP_FRACTIONS) - 1, fbin))
        ships = int(math.floor(src["ships"] * _SHIP_FRACTIONS[fbin]))
        if ships < _MIN_SHIPS:
            continue
        # For non-owned targets: fleet must be large enough to capture
        if tgt["owner"] != player and ships <= tgt["ships"]:
            continue

        angle = math.atan2(tgt["y"] - src["y"], tgt["x"] - src["x"])
        moves.append([src["id"], angle, ships])

    return moves
