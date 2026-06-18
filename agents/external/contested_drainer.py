"""Contested non-producer FIXTURE bot — a *strong* half-drain planner.

Purpose: the existing hand-built archetypes (``half_drainer``, ``swarmer``) lose to v5
**100%** — too weak to be a *contested* instrument. To measure whether v5 has exploitable
structure against a genuinely NON-producer style (rather than re-measuring the producer
mirror), we need a partial-send opponent strong enough that v5 wins only ~50-70%. The
replay findings asserted hand-built bots "can't reach producer-tier," but that was tested
only on the bare ``archetype_common`` planner, which has no consolidation, no defense, and
no global value — it is weak for *competence* reasons, not because half-drain is inherently
weak (Isaiah @ Tufa, LB #1, half-drains and is the single strongest agent).

This planner keeps the **partial-send (half-drain) signature** — attack fleets are sized
*just to capture* (median send-fraction ~0.5, the structural delta a producer self-model
cannot predict) — but adds the macro competence the simple archetype lacks:

  1. **Consolidation** — rear (non-frontier) garrisons stream forward to the most-pressured
     frontier planet, so ships are not stranded and attacks/defense are affordable.
  2. **Defense** — incoming enemy fleets are detected (ray closest-approach); a threatened
     planet's garrison is reserved (not over-drained) and reinforced from neighbors.
  3. **Production-first target value** — captures are ranked by production/compounding,
     affordability (small fraction of garrison), and distance.
  4. **Flight-reinforcement awareness** — enemy targets are sized for production accrued
     during the fleet's flight (so doomed under-sends are avoided).

NOT a competitor and NOT a vendored public agent — a deterministic MEASUREMENT FIXTURE.
Strength is tunable via ``CONFIG`` so the off-mirror gate can be placed in the contested
band. Obs parsing follows the project rule (attribute -> dict -> positional list).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SUN_X, SUN_Y, SUN_R = 50.0, 50.0, 10.0
_SUN_MARGIN = 11.5
_MAX_SPEED = 6.0

_P = {"id": 0, "owner": 1, "x": 2, "y": 3, "radius": 4, "ships": 5, "production": 6}
_F = {"id": 0, "owner": 1, "x": 2, "y": 3, "angle": 4, "from_planet_id": 5, "ships": 6}


@dataclass(frozen=True)
class Config:
    """Strength / style knobs (tune to land v5 in the contested 50-70% band)."""

    # --- attack (the half-drain signature) ---
    targets_per_source: int = 3        # captures launched per frontier source per turn
    drain_cap: float = 0.85            # max fraction of a source's garrison spent on attacks
    capture_margin: float = 2.0        # ships over the capture floor (slack for combat)
    min_ships_to_attack: float = 3.0   # open from the home planet immediately
    min_fleet: int = 3
    dist_weight: float = 0.05          # prefer NEAR targets (tempo) in the capture score
    # --- consolidation (internal logistics; full sends, NOT the attack fingerprint) ---
    consolidate: bool = True
    rear_keep: float = 1.0             # ships a rear planet keeps when forwarding surplus
    frontier_dist: float = 45.0        # a non-owned planet within this => source is frontier
    # --- defense ---
    defend: bool = True
    defense_margin: float = 1.0        # extra garrison held over projected incoming threat
    defense_cap_frac: float = 0.7      # never reserve more than this fraction (always attack)


CONFIG = Config()


# --- obs parsing ------------------------------------------------------------

def _field(entry, name: str, idx: int):
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


def _fleets(obs) -> list[dict]:
    out = []
    for f in _obs_get(obs, "fleets", []) or []:
        out.append(
            {
                "owner": int(_field(f, "owner", _F["owner"])),
                "x": float(_field(f, "x", _F["x"])),
                "y": float(_field(f, "y", _F["y"])),
                "angle": float(_field(f, "angle", _F["angle"])),
                "ships": float(_field(f, "ships", _F["ships"])),
            }
        )
    return out


# --- geometry ---------------------------------------------------------------

def _fleet_speed(ships: float) -> float:
    if ships <= 1.0:
        return 1.0
    return 1.0 + (_MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000.0)) ** 1.5


def _seg_dist_to_sun(ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 <= 1e-9:
        return math.hypot(SUN_X - ax, SUN_Y - ay)
    t = ((SUN_X - ax) * dx + (SUN_Y - ay) * dy) / l2
    t = max(0.0, min(1.0, t))
    px, py = ax + t * dx, ay + t * dy
    return math.hypot(SUN_X - px, SUN_Y - py)


def _lead_position(t: dict, eta: float, ang_vel: float) -> tuple[float, float]:
    rx, ry = t["x"] - SUN_X, t["y"] - SUN_Y
    rc = math.hypot(rx, ry)
    if rc + t["radius"] >= 50.0 or ang_vel == 0.0:
        return t["x"], t["y"]
    a = math.atan2(ry, rx) + ang_vel * eta
    return SUN_X + rc * math.cos(a), SUN_Y + rc * math.sin(a)


def _aim(src: dict, t: dict, ang_vel: float, *, est_ships: float):
    """Return (eta, angle, tx, ty) leading an orbiting target, or None if sun-blocked."""
    tx, ty = t["x"], t["y"]
    for _ in range(2):
        dist = max(1.0, math.hypot(tx - src["x"], ty - src["y"]) - src["radius"] - t["radius"])
        eta = dist / _fleet_speed(max(est_ships, 8.0))
        tx, ty = _lead_position(t, eta, ang_vel)
    if _seg_dist_to_sun(src["x"], src["y"], tx, ty) <= _SUN_MARGIN:
        return None
    dist = max(1.0, math.hypot(tx - src["x"], ty - src["y"]) - src["radius"] - t["radius"])
    eta = dist / _fleet_speed(max(est_ships, 8.0))
    angle = math.atan2(ty - src["y"], tx - src["x"])
    return eta, angle, tx, ty


def _incoming_threat(planet: dict, fleets: list[dict], pid: int) -> float:
    """Enemy ships on fleets whose straight path will strike ``planet`` (ray closest-approach).

    Approximate but cheap: a fleet at (fx,fy) heading ``angle`` threatens the planet if its
    forward ray passes within (radius + slack) of the planet center. Sums those ships.
    """
    px, py, r = planet["x"], planet["y"], planet["radius"]
    total = 0.0
    for f in fleets:
        if f["owner"] == pid or f["owner"] < 0:
            continue
        ux, uy = math.cos(f["angle"]), math.sin(f["angle"])
        proj = (px - f["x"]) * ux + (py - f["y"]) * uy
        if proj <= 0.0:
            continue  # heading away
        cx, cy = f["x"] + proj * ux, f["y"] + proj * uy
        if math.hypot(px - cx, py - cy) <= r + 2.0:
            total += f["ships"]
    return total


# --- planner ----------------------------------------------------------------

def plan(obs, config: Config = CONFIG) -> list:
    pid = int(_obs_get(obs, "player", 0))
    ang_vel = float(_obs_get(obs, "angular_velocity", 0.0) or 0.0)
    planets = _planets(obs)
    if not planets:
        return []
    fleets = _fleets(obs)
    owned = [p for p in planets if p["owner"] == pid]
    non_owned = [p for p in planets if p["owner"] != pid]
    if not owned or not non_owned:
        return []

    # Per-owned-planet defensive reservation: hold enough to survive projected incoming.
    reserve: dict[int, float] = {}
    for p in owned:
        if config.defend:
            threat = _incoming_threat(p, fleets, pid)
            r = max(0.0, threat + config.defense_margin) if threat > 0 else 0.0
            # never strangle a source: hold at most defense_cap_frac of the garrison so the
            # rest can always attack (over-reserving = the Cluster-9/12 passivity trap).
            reserve[p["id"]] = min(r, config.defense_cap_frac * p["ships"])
        else:
            reserve[p["id"]] = 0.0

    # Frontier classification: a source is "frontier" if a non-owned planet is near it
    # OR it faces incoming threat. Rear planets forward their surplus to the frontier.
    def is_frontier(p: dict) -> bool:
        if config.defend and _incoming_threat(p, fleets, pid) > 0:
            return True
        for t in non_owned:
            if math.hypot(t["x"] - p["x"], t["y"] - p["y"]) <= config.frontier_dist:
                return True
        return False

    frontier = [p for p in owned if is_frontier(p)]
    rear = [p for p in owned if p not in frontier]

    moves: list = []
    # available budget per source after defensive reservation (mutated as we spend)
    avail = {p["id"]: max(0.0, p["ships"] - reserve[p["id"]]) for p in owned}

    # --- 1. attacks from frontier planets (capture-minimal = half-drain signature) ---
    attack_sources = sorted(frontier, key=lambda p: -p["ships"])
    for src in attack_sources:
        if avail[src["id"]] < config.min_ships_to_attack:
            continue
        cands = []
        for t in non_owned:
            aimed = _aim(src, t, ang_vel, est_ships=max(t["ships"], 8.0))
            if aimed is None:
                continue
            eta, angle, _, _ = aimed
            # capture floor: neutrals don't produce until owned; enemies reinforce in flight.
            if t["owner"] == -1:
                need = t["ships"] + config.capture_margin
            else:
                need = t["ships"] + t["prod"] * eta + config.capture_margin
            need = math.ceil(need)
            if need > src["ships"]:
                continue
            frac = need / max(src["ships"], 1.0)
            dist = math.hypot(t["x"] - src["x"], t["y"] - src["y"])
            # prefer high production, then cheap (small fraction), then NEAR (tempo).
            score = -3.0 * t["prod"] + frac + 0.02 * need + config.dist_weight * dist
            cands.append((score, need, angle))
        if not cands:
            continue
        cands.sort(key=lambda c: c[0])

        spent = 0.0
        launched = 0
        cap = config.drain_cap * src["ships"]
        for i, (_, need, angle) in enumerate(cands):
            if launched >= config.targets_per_source:
                break
            # first capture always allowed (decisive opening); later ones respect the cap.
            if i > 0 and spent + need > cap:
                continue
            if spent + need > avail[src["id"]]:
                continue
            ships_w = max(config.min_fleet, int(need))
            if ships_w > avail[src["id"]] - spent:
                ships_w = int(avail[src["id"]] - spent)
            if ships_w < config.min_fleet:
                continue
            moves.append([src["id"], float(angle), int(ships_w)])
            spent += ships_w
            launched += 1
        avail[src["id"]] -= spent

    # --- 2. consolidation: rear surplus streams to the most-pressured frontier ---
    if config.consolidate and frontier:
        for src in rear:
            surplus = src["ships"] - config.rear_keep
            if surplus < config.min_fleet:
                continue
            # nearest frontier planet reachable without crossing the sun.
            best = None
            best_d = float("inf")
            for fpl in frontier:
                if fpl["id"] == src["id"]:
                    continue
                aimed = _aim(src, fpl, ang_vel, est_ships=surplus)
                if aimed is None:
                    continue
                _, angle, tx, ty = aimed
                d = math.hypot(fpl["x"] - src["x"], fpl["y"] - src["y"])
                if d < best_d:
                    best_d, best = d, angle
            if best is None:
                continue
            ships_w = int(min(surplus, src["ships"]))
            if ships_w >= config.min_fleet:
                moves.append([src["id"], float(best), int(ships_w)])

    return moves


def agent(obs, config=None) -> list:
    return plan(obs, CONFIG)
