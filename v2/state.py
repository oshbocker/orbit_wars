"""Fleet destination prediction for V2 pipeline."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.features import (
    BOARD_SIZE,
    SUN_RADIUS,
    fleet_hits_planet,
    fleet_speed,
    passes_through_sun,
    planet_pos_at,
)
from src.game_types import FleetState, GameState, PlanetState, SUN_X, SUN_Y


@dataclass(slots=True)
class IncomingFleetInfo:
    """Per-planet incoming fleet aggregation, indexed by relative team (0=own, 1-3=enemies)."""
    ships: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    eta: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])


def predict_fleet_destination(
    fleet: FleetState,
    planets: list[PlanetState],
    step: int,
    angular_velocity: float,
) -> tuple[PlanetState | None, float]:
    """Predict which planet a fleet will hit.

    Static planets: ray-circle intersection via fleet_hits_planet().
    Orbiting planets: step-by-step forward simulation.
    Returns (target_planet, eta_turns) or (None, inf) if no hit.
    """
    best_planet: PlanetState | None = None
    best_eta = float("inf")

    speed = fleet_speed(fleet.ships)

    for planet in planets:
        if not planet.is_orbiting:
            # Static planet: direct ray-circle check
            eta = fleet_hits_planet(fleet, planet)
            if eta is not None and eta < best_eta:
                best_eta = eta
                best_planet = planet
        else:
            # Orbiting planet: step-by-step forward sim
            eta = _orbiting_hit_check(fleet, planet, step, angular_velocity, speed)
            if eta is not None and eta < best_eta:
                best_eta = eta
                best_planet = planet

    if best_planet is None:
        return None, float("inf")

    # Check if fleet hits sun before reaching the planet
    fx, fy = fleet.x, fleet.y
    dx = math.cos(fleet.angle) * speed
    dy = math.sin(fleet.angle) * speed
    hit_x = fx + dx * best_eta
    hit_y = fy + dy * best_eta
    if passes_through_sun(fx, fy, hit_x, hit_y):
        return None, float("inf")

    return best_planet, best_eta


def _orbiting_hit_check(
    fleet: FleetState,
    planet: PlanetState,
    step: int,
    angular_velocity: float,
    speed: float,
    max_steps: int = 100,
) -> float | None:
    """Check if fleet collides with an orbiting planet via forward sim."""
    fx, fy = fleet.x, fleet.y
    dx = math.cos(fleet.angle) * speed
    dy = math.sin(fleet.angle) * speed
    hit_r = planet.radius + 0.5

    for t in range(1, max_steps + 1):
        # Fleet position at time t
        new_fx = fx + dx * t
        new_fy = fy + dy * t

        # Out of bounds check
        if new_fx < 0 or new_fx > BOARD_SIZE or new_fy < 0 or new_fy > BOARD_SIZE:
            return None

        # Planet position at step + t
        px, py = planet_pos_at(planet, step + t, angular_velocity)

        # Check collision: continuous segment from (prev_fx, prev_fy) to (new_fx, new_fy)
        prev_fx = fx + dx * (t - 1)
        prev_fy = fy + dy * (t - 1)
        if _segment_circle_hit(prev_fx, prev_fy, new_fx, new_fy, px, py, hit_r):
            return float(t)

    return None


def _segment_circle_hit(
    x1: float, y1: float, x2: float, y2: float,
    cx: float, cy: float, r: float,
) -> bool:
    """Check if line segment (x1,y1)→(x2,y2) intersects circle (cx,cy,r)."""
    sdx, sdy = x2 - x1, y2 - y1
    fx, fy = x1 - cx, y1 - cy
    a = sdx * sdx + sdy * sdy
    if a < 1e-10:
        return fx * fx + fy * fy <= r * r
    b = 2 * (fx * sdx + fy * sdy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 < t2)


def compute_incoming_fleets(
    state: GameState,
    player: int,
) -> dict[int, IncomingFleetInfo]:
    """For each planet, aggregate all incoming fleets by relative team.

    Team 0 = player's own fleets
    Teams 1-3 = enemies (ordered by first encounter)
    """
    result: dict[int, IncomingFleetInfo] = {}
    enemy_ids: list[int] = []

    for fleet in state.fleets:
        target, eta = predict_fleet_destination(
            fleet, state.planets, state.step, state.angular_velocity,
        )
        if target is None:
            continue

        # Determine relative team index
        if fleet.owner == player:
            team = 0
        else:
            if fleet.owner not in enemy_ids:
                enemy_ids.append(fleet.owner)
            idx = enemy_ids.index(fleet.owner)
            team = min(idx + 1, 3)  # cap at 3

        if target.id not in result:
            result[target.id] = IncomingFleetInfo()
        info = result[target.id]

        # Weighted average ETA
        old_ships = info.ships[team]
        new_total = old_ships + fleet.ships
        if new_total > 0:
            info.eta[team] = (info.eta[team] * old_ships + eta * fleet.ships) / new_total
        info.ships[team] = new_total

    return result
