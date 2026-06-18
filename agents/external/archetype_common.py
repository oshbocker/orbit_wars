"""Shared planner for the off-mirror archetype FIXTURE bots (half_drainer, swarmer).

These are NOT competitors and NOT vendored public agents — they are deterministic,
rule-based MEASUREMENT FIXTURES that play structurally NON-producer styles, so we can
test whether v5's opponent-injection hook (``opp_inject_waves``) buys win-rate against
opponents the producer-self-model in Cluster 11 could not see. Their structural params
are seeded from the top-tier replay diagnostic (``top-tier-replay-diagnostic`` memory /
``TOP_TIER_REPLAY_CORPUS.md``):

    half-drain (Isaiah @ Tufa, LB #1 1762): median send-fraction 0.52, ~1.3-1.6 waves/turn
    swarm     (213tubo, 1536):              median send-fraction 0.67, ~14.2 waves/turn

The non-producer signature is **capture-minimal sizing**: producer/v5 *full-drain*
(send the whole garrison, ~95% of sends are the full safe_drain), whereas these send
only enough to take the target and keep the rest in reserve (partial sends) — the
half-drainer takes one cheap target per source per turn (low wave count), the swarmer
spreads many small capture fleets across many targets (high wave count). Capture-aware
sizing is also what makes them credible opponents (a fixed-fraction send can never
capture a 9-27 ship neutral from a 10-ship home, so it never expands).

Obs parsing follows the project rule (and ``scripts/arena.py:_field``): try attribute
access first (Kaggle ``Struct``), then dict, then positional list — NEVER
``isinstance(list/tuple)`` first (Structs are iterable over keys, not values).
"""

from __future__ import annotations

import math

SUN_X, SUN_Y, SUN_R = 50.0, 50.0, 10.0
# Require the source->target segment to clear the sun by this margin. The fleet spawns
# just outside the source radius and travels straight; if the segment to the target
# clears the sun, the fleet reaches the target (continuous collision) before any sun
# crossing. Margin covers the spawn offset + numerical slack.
_SUN_MARGIN = 11.5
_MAX_SPEED = 6.0

# Planet layout: [id, owner, x, y, radius, ships, production]
_P = {"id": 0, "owner": 1, "x": 2, "y": 3, "radius": 4, "ships": 5, "production": 6}


def _field(entry, name: str, idx: int):
    """Struct attribute -> dict key -> positional index (the project's parse order)."""
    if hasattr(entry, name):
        return getattr(entry, name)
    if isinstance(entry, dict):
        return entry[name]
    return entry[idx]


def _obs_get(obs, name: str, default):
    if hasattr(obs, name):
        return getattr(obs, name)
    if isinstance(obs, dict):
        return obs.get(name, default)
    return default


def _planets(obs) -> list[dict]:
    out = []
    for p in _obs_get(obs, "planets", []) or []:
        out.append(
            {
                "id": int(_field(p, "id", _P["id"])),
                "owner": int(_field(p, "owner", _P["owner"])),
                "x": float(_field(p, "x", _P["x"])),
                "y": float(_field(p, "y", _P["y"])),
                "radius": float(_field(p, "radius", _P["radius"])),
                "ships": float(_field(p, "ships", _P["ships"])),
                "prod": float(_field(p, "production", _P["production"])),
            }
        )
    return out


def _seg_dist_to_sun(ax: float, ay: float, bx: float, by: float) -> float:
    """Minimum distance from segment AB to the sun center."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 <= 1e-9:
        return math.hypot(SUN_X - ax, SUN_Y - ay)
    t = ((SUN_X - ax) * dx + (SUN_Y - ay) * dy) / l2
    t = max(0.0, min(1.0, t))
    px, py = ax + t * dx, ay + t * dy
    return math.hypot(SUN_X - px, SUN_Y - py)


def _fleet_speed(ships: float) -> float:
    if ships <= 1.0:
        return 1.0
    return 1.0 + (_MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5


def _lead_position(t: dict, eta: float, ang_vel: float) -> tuple[float, float]:
    """Predict an orbiting (inner) planet's position ``eta`` turns ahead.

    Inner planets rotate about the sun at the global ``angular_velocity``; outer
    (static) planets don't. Detect inner by ``dist_center + radius < 50`` and rotate
    the current position by ``ang_vel * eta`` about the center.
    """
    rx, ry = t["x"] - SUN_X, t["y"] - SUN_Y
    rc = math.hypot(rx, ry)
    if rc + t["radius"] >= 50.0 or ang_vel == 0.0:
        return t["x"], t["y"]  # static
    a = math.atan2(ry, rx) + ang_vel * eta
    return SUN_X + rc * math.cos(a), SUN_Y + rc * math.sin(a)


def _capture_plan(src: dict, t: dict, ang_vel: float):
    """Return (need, angle) to capture ``t`` from ``src``, or None if not affordable.

    ``need`` = target garrison + (enemy production during flight) + margin, sized just
    to capture (capture-minimal). Leads orbiting targets. Skips sun-crossing shots.
    """
    # First pass: aim at current position to estimate eta, then lead and re-aim.
    tx, ty = t["x"], t["y"]
    for _ in range(2):
        dist = max(1.0, math.hypot(tx - src["x"], ty - src["y"]) - src["radius"] - t["radius"])
        # provisional fleet ~ target garrison sets the speed estimate
        eta = dist / _fleet_speed(max(t["ships"], 8.0))
        tx, ty = _lead_position(t, eta, ang_vel)
    if _seg_dist_to_sun(src["x"], src["y"], tx, ty) <= _SUN_MARGIN:
        return None
    dist = max(1.0, math.hypot(tx - src["x"], ty - src["y"]) - src["radius"] - t["radius"])
    eta = dist / _fleet_speed(max(t["ships"], 8.0))
    # Neutrals don't produce until owned; enemy planets reinforce during the flight.
    need = t["ships"] + 1.0 if t["owner"] == -1 else t["ships"] + t["prod"] * eta + 2.0
    angle = math.atan2(ty - src["y"], tx - src["x"])
    return math.ceil(need), angle


def plan(
    obs,
    *,
    targets_per_source: int,
    max_total_waves: int,
    drain_cap: float,
    min_ships: float,
    min_fleet: int,
    split_k: int = 1,
) -> list:
    """Generic capture-minimal archetype planner shared by the fixtures.

    For owned planets (most ships first), find the cheapest-to-capture NON-owned
    targets (production-biased) and launch capture-sized fleets at up to
    ``targets_per_source`` of them. The source always gets its single best capture
    (so it can open from a 10-ship home); additional waves fire only while cumulative
    outflow stays under ``drain_cap`` of the garrison (the reserve discipline). Capped
    at ``max_total_waves`` launches for the whole turn. Returns Kaggle moves
    ``[[from_planet_id, angle_radians, num_ships], ...]``.

    ``split_k`` > 1 emits each capture as ``k`` equal same-target sub-fleets (identical
    source/angle/size => identical speed => same arrival => they aggregate in combat,
    so the capture still succeeds) — this is the SWARM signature (many small waves)
    without sacrificing the ability to capture, which naive many-tiny-fleets cannot.
    """
    pid = int(_obs_get(obs, "player", 0))
    ang_vel = float(_obs_get(obs, "angular_velocity", 0.0) or 0.0)
    planets = _planets(obs)
    if not planets:
        return []
    owned = sorted(
        (p for p in planets if p["owner"] == pid and p["ships"] >= min_ships),
        key=lambda p: -p["ships"],
    )
    non_owned = [p for p in planets if p["owner"] != pid]
    if not owned or not non_owned:
        return []

    moves: list = []
    for src in owned:
        if len(moves) >= max_total_waves:
            break
        # Score every affordable capture: prefer high production, then cheap (small
        # fraction of our garrison), then near.
        cands = []
        for t in non_owned:
            cp = _capture_plan(src, t, ang_vel)
            if cp is None:
                continue
            need, angle = cp
            if need > src["ships"]:
                continue  # can't take it from here this turn
            frac = need / max(src["ships"], 1.0)
            score = -3.0 * t["prod"] + frac + 0.02 * need
            cands.append((score, need, angle))
        if not cands:
            continue
        cands.sort(key=lambda c: c[0])

        spent = 0.0
        launched = 0
        for i, (_, need, angle) in enumerate(cands):
            if launched >= targets_per_source or len(moves) >= max_total_waves:
                break
            # First wave is always allowed (decisive opening). Later waves keep reserve.
            if i > 0 and spent + need > drain_cap * src["ships"]:
                continue
            if spent + need > src["ships"]:
                continue
            ships_w = max(min_fleet, int(need))
            if spent + ships_w > src["ships"]:
                ships_w = int(src["ships"] - spent)
            if ships_w < min_fleet:
                break
            # Split into k equal same-target sub-fleets that aggregate on arrival. Round
            # each up so k*sub >= ships_w (still captures); never below min_fleet, never
            # more than k fleets, and never overspend the garrison.
            k = max(1, split_k)
            sub = max(min_fleet, -(-ships_w // k))  # ceil(ships_w / k)
            for _ in range(k):
                if len(moves) >= max_total_waves or spent + sub > src["ships"]:
                    break
                moves.append([src["id"], float(angle), int(sub)])
                spent += sub
            launched += 1

    return moves
