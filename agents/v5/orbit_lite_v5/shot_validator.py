"""Reject-only shot validator (konbu17 pattern, LEADERBOARD_CLIMB_PLAN Phase 2.1).

A ~2.4K-param numpy MLP (24 -> 64 -> 32 -> 1, sigmoid) scores every emitted
attack shot with P("we own the target within [arrival, arrival+10]"); shots
below the veto threshold are dropped. Own-planet reinforcements are exempt, so
the wrapper can only remove bad attacks from the base planner — fail-safe by
construction.

The SAME encoder runs at label-harvest time (scripts/harvest_shots.py) and at
inference, so the feature distribution matches exactly. Feature layout is
copied verbatim from the public konbu17 validator (agents/external/
shot_validator_hybrid.py) whose +19pp local result we are replicating on v5.

Planet/fleet rows are indexed positionally (real Kaggle env rows; fast_env dict
rows are normalized upstream by agents.load_named_agent).
"""

from __future__ import annotations

import math

import numpy as np

BOARD = 100.0
MAX_SPEED = 6.0
FEATURE_DIM = 24
DEFAULT_THRESHOLD = 0.4


class NumpyValidator:
    """MLP forward pass on the npz weight layout {w0,b0,w2,b2,w4,b4}."""

    def __init__(self, npz_path):
        npz = np.load(str(npz_path))
        self.w0 = npz["w0"]
        self.b0 = npz["b0"]
        self.w2 = npz["w2"]
        self.b2 = npz["b2"]
        self.w4 = npz["w4"]
        self.b4 = npz["b4"]

    def proba(self, x: np.ndarray) -> np.ndarray:
        h = np.maximum(0.0, x @ self.w0.T + self.b0)
        h = np.maximum(0.0, h @ self.w2.T + self.b2)
        z = (h @ self.w4.T + self.b4).reshape(-1)
        return 1.0 / (1.0 + np.exp(-z))


def find_target_ray(src_xy, send_angle, planets, ray_horizon=200.0, perp_margin=1.0):
    """Planet id the shot is aimed at (smallest perpendicular miss along the ray)."""
    sx, sy = src_xy
    fx = math.cos(send_angle)
    fy = math.sin(send_angle)
    best_pid = -1
    best_perp = 1e9
    for p in planets:
        pid = int(p[0])
        px = float(p[2])
        py = float(p[3])
        pr = float(p[4])
        dx = px - sx
        dy = py - sy
        t = dx * fx + dy * fy
        if t <= 0 or t > ray_horizon:
            continue
        perp = abs(dx * fy - dy * fx)
        if perp <= pr + perp_margin and perp < best_perp:
            best_perp = perp
            best_pid = pid
    return best_pid


def shot_eta(src, tgt, ships_sent) -> float:
    """ETA in turns under the engine speed formula (boundary-to-boundary)."""
    _, sx, sy, sr = int(src[1]), float(src[2]), float(src[3]), float(src[4])
    tx, ty, tr = float(tgt[2]), float(tgt[3]), float(tgt[4])
    dist = max(math.hypot(tx - sx, ty - sy) - sr - tr, 0.0)
    if ships_sent <= 0:
        speed = 1.0
    else:
        speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(ships_sent, 1)) / math.log(1000.0)) ** 1.5
    return dist / max(speed, 0.5)


def encode_shot(obs, src_id: int, target_id: int, ships_sent: int):
    """24-dim feature vector for one shot, or None if src/target are unknown."""
    pdict = {}
    for p in obs["planets"]:
        pid = int(p[0])
        pdict[pid] = (int(p[1]), float(p[2]), float(p[3]), float(p[4]), int(p[5]), float(p[6]))
    if src_id not in pdict or target_id not in pdict:
        return None
    src = pdict[src_id]
    tgt = pdict[target_id]
    me = int(obs.get("player", 0))
    fleets = obs.get("fleets", [])
    planets = obs["planets"]
    my_ships_total = sum(int(p[5]) for p in planets if int(p[1]) == me)
    enemy_ships_total = sum(int(p[5]) for p in planets if int(p[1]) >= 0 and int(p[1]) != me)
    my_planets = sum(1 for p in planets if int(p[1]) == me)
    enemy_planets = sum(1 for p in planets if int(p[1]) >= 0 and int(p[1]) != me)
    src_owner, sx, sy, sr, ss, sp = src
    tgt_owner, tx, ty, tr, ts, tp = tgt
    dist = max(math.hypot(tx - sx, ty - sy) - sr - tr, 0.0)
    if ships_sent <= 0:
        speed = 1.0
    else:
        speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(ships_sent, 1)) / math.log(1000.0)) ** 1.5
    eta = dist / max(speed, 0.5)
    own_self = 1.0 if tgt_owner == me else 0.0
    own_neutral = 1.0 if tgt_owner < 0 else 0.0
    own_enemy = 1.0 if (tgt_owner >= 0 and tgt_owner != me) else 0.0
    ship_frac = ships_sent / max(ss, 1)
    ally_n = 0
    ally_s = 0
    enemy_n = 0
    enemy_s = 0
    for f in fleets:
        owner = int(f[1])
        shp = int(f[6])
        if owner == me:
            ally_n += 1
            ally_s += shp
        else:
            enemy_n += 1
            enemy_s += shp
    turn = int(obs.get("step", 0))
    return np.array(
        [
            ss / 100.0, sp / 5.0, sr / 4.0,
            ts / 100.0, tp / 5.0, tr / 4.0,
            own_self, own_neutral, own_enemy,
            ships_sent / 100.0, ship_frac,
            dist / BOARD, eta / 60.0, speed / MAX_SPEED,
            ally_n / 10.0, ally_s / 100.0,
            enemy_n / 10.0, enemy_s / 100.0,
            turn / 500.0,
            my_ships_total / 200.0, enemy_ships_total / 200.0,
            (my_ships_total - enemy_ships_total) / 200.0,
            my_planets / 20.0, enemy_planets / 20.0,
        ],
        dtype=np.float32,
    )


def apply_veto(moves, obs, validator: NumpyValidator, threshold: float):
    """Drop attack shots with P(success) < threshold; keep everything else.

    Own-planet reinforcements and shots whose target the ray cast cannot
    identify (e.g. aimed at a predicted future position of an orbiting planet)
    are always kept.
    """
    if not moves or validator is None:
        return moves
    side = int(obs.get("player", 0))
    planets = obs["planets"]
    owner_by_id = {}
    src_xy = {}
    for p in planets:
        pid = int(p[0])
        owner_by_id[pid] = int(p[1])
        src_xy[pid] = (float(p[2]), float(p[3]))
    feats = []
    idxs = []
    for i, mv in enumerate(moves):
        try:
            src_id = int(mv[0])
            ang = float(mv[1])
            ships = int(mv[2])
        except (TypeError, ValueError, IndexError):
            continue
        if src_id not in src_xy:
            continue
        tgt_id = find_target_ray(src_xy[src_id], ang, planets)
        if tgt_id < 0 or tgt_id == src_id:
            continue
        if owner_by_id.get(tgt_id, -2) == side:
            continue  # own-planet reinforcement: always keep
        feat = encode_shot(obs, src_id, tgt_id, ships)
        if feat is None:
            continue
        feats.append(feat)
        idxs.append(i)
    if not feats:
        return moves
    probs = validator.proba(np.stack(feats))
    keep = [True] * len(moves)
    for i, prob in zip(idxs, probs, strict=False):
        if prob < threshold:
            keep[i] = False
    return [mv for i, mv in enumerate(moves) if keep[i]]
