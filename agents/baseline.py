"""
Deterministic baseline agent for Orbit Wars.

Strategy:
  - For each owned planet, find the best target (weakest nearby enemy/neutral).
  - Account for estimated travel time when computing required ships.
  - Avoid routing fleets through the sun.
  - Deduplicate targets so multiple planets don't pile onto the same target.
  - Hold back a minimum garrison to defend against incoming threats.

This agent is the benchmark all RL agents must beat.
"""

import math
from typing import Any

# Sun is at (50, 50) with radius 10; give it a small safety buffer.
SUN_X, SUN_Y, SUN_RADIUS = 50.0, 50.0, 10.0
SUN_SAFE_RADIUS = SUN_RADIUS + 2.0

MAX_SHIP_SPEED = 6.0
MIN_GARRISON = 5  # ships to keep on each planet for defense


def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _fleet_speed(ships: int, max_speed: float = MAX_SHIP_SPEED) -> float:
    if ships <= 1:
        return 1.0
    return 1.0 + (max_speed - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def _travel_time(src_x, src_y, dst_x, dst_y, ships: int) -> float:
    d = _dist(src_x, src_y, dst_x, dst_y)
    return d / _fleet_speed(ships)


def _passes_through_sun(x1: float, y1: float, x2: float, y2: float) -> bool:
    """Return True if the straight line from (x1,y1) to (x2,y2) passes within
    SUN_SAFE_RADIUS of the sun center."""
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - SUN_X, y1 - SUN_Y
    a = dx * dx + dy * dy
    if a == 0:
        return False
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - SUN_SAFE_RADIUS ** 2
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return False
    t1 = (-b - math.sqrt(discriminant)) / (2 * a)
    t2 = (-b + math.sqrt(discriminant)) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 < t2)


def _ships_needed(target, travel_turns: float) -> int:
    """Ships needed to capture a target planet, accounting for production during travel."""
    production_during_flight = target.production * travel_turns if target.owner >= 0 else 0
    return int(target.ships + production_during_flight) + 1


def agent(obs, config=None) -> list:
    """
    Orbit Wars agent function.  Accepts both dict and namespace observations.
    Returns list of [from_planet_id, angle_radians, num_ships] moves.
    """
    # ── parse observation ──────────────────────────────────────────────────
    if isinstance(obs, dict):
        player = obs.get("player", 0)
        raw_planets = obs.get("planets", [])
        raw_fleets = obs.get("fleets", [])
    else:
        player = obs.player
        raw_planets = obs.planets
        raw_fleets = obs.fleets

    # Build simple Planet namedtuple-like objects from raw lists
    planets = [_Planet(*p) for p in raw_planets]
    fleets = [_Fleet(*f) for f in raw_fleets]

    my_planets = [p for p in planets if p.owner == player]
    other_planets = [p for p in planets if p.owner != player]

    if not other_planets or not my_planets:
        return []

    # ── compute incoming threat per planet ────────────────────────────────
    incoming: dict[int, int] = {}
    for f in fleets:
        if f.owner != player:
            # approximate: fleet is near which planet?
            for p in my_planets:
                if _dist(f.x, f.y, p.x, p.y) < 5.0:
                    incoming[p.id] = incoming.get(p.id, 0) + f.ships

    # ── score each potential target (lower is better) ─────────────────────
    def target_score(src, tgt):
        d = _dist(src.x, src.y, tgt.x, tgt.y)
        # penalise sun-crossing routes heavily
        sun_penalty = 1000 if _passes_through_sun(src.x, src.y, tgt.x, tgt.y) else 0
        # prefer weaker, closer, higher-production targets
        return d + tgt.ships * 0.5 - tgt.production * 10 + sun_penalty

    moves = []
    claimed: set[int] = set()  # target planet ids already claimed this turn

    # Sort own planets by production descending (prioritise strong planets)
    for mine in sorted(my_planets, key=lambda p: -p.production):
        available = mine.ships - MIN_GARRISON - incoming.get(mine.id, 0)
        if available < 2:
            continue

        # Find best unclaimed target
        candidates = [t for t in other_planets if t.id not in claimed]
        if not candidates:
            break

        best = min(candidates, key=lambda t: target_score(mine, t))

        if _passes_through_sun(mine.x, mine.y, best.x, best.y):
            # Filter sun-crossing targets entirely
            candidates_safe = [t for t in candidates
                                if not _passes_through_sun(mine.x, mine.y, t.x, t.y)]
            if not candidates_safe:
                continue
            best = min(candidates_safe, key=lambda t: target_score(mine, t))

        ships_to_send = _ships_needed(
            best,
            _travel_time(mine.x, mine.y, best.x, best.y, min(available, 100))
        )
        ships_to_send = min(ships_to_send, available)

        if ships_to_send < 2:
            continue

        angle = math.atan2(best.y - mine.y, best.x - mine.x)
        moves.append([mine.id, angle, ships_to_send])
        claimed.add(best.id)

    return moves


# ── lightweight data classes (avoid importing kaggle_environments at import time) ─

class _Planet:
    __slots__ = ("id", "owner", "x", "y", "radius", "ships", "production")

    def __init__(self, id, owner, x, y, radius, ships, production):
        self.id = id
        self.owner = owner
        self.x = x
        self.y = y
        self.radius = radius
        self.ships = ships
        self.production = production


class _Fleet:
    __slots__ = ("id", "owner", "x", "y", "angle", "from_planet_id", "ships")

    def __init__(self, id, owner, x, y, angle, from_planet_id, ships):
        self.id = id
        self.owner = owner
        self.x = x
        self.y = y
        self.angle = angle
        self.from_planet_id = from_planet_id
        self.ships = ships


if __name__ == "__main__":
    # Quick smoke test using kaggle_environments
    from kaggle_environments import make
    env = make("orbit_wars", debug=True)
    env.run([agent, "random"])
    final = env.steps[-1]
    for i, s in enumerate(final):
        print(f"Player {i}: reward={s.reward}, status={s.status}")
