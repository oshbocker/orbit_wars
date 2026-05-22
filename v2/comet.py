"""Comet evacuation handler for V2 pipeline."""
from __future__ import annotations

import math
from typing import Any

from src.game_types import GameState, PlanetState, SUN_X, SUN_Y


def comet_evacuation_moves(
    state: GameState,
    comet_planet_ids: list[int] | None,
    obs: Any,
) -> tuple[list[list[float | int]], set[int]]:
    """If we own a comet about to leave the board, evacuate ships.

    Returns (moves, evacuated_planet_ids) where moves is a list of
    [planet_id, angle, ships] and evacuated_planet_ids are planet IDs
    that were evacuated (should be excluded from RL decisions).
    """
    if not comet_planet_ids:
        return [], set()

    moves: list[list[float | int]] = []
    evacuated: set[int] = set()

    # Parse comet path data from obs if available
    comets_data = _get_comets_data(obs)
    if comets_data is None:
        return [], set()

    for pid in comet_planet_ids:
        planet = state.planets_by_id.get(pid)
        if planet is None or planet.owner != state.player or planet.ships <= 0:
            continue

        # Check if comet is about to exit
        if not _comet_near_exit(pid, comets_data):
            continue

        # Find nearest non-comet owned planet
        target = _nearest_owned_planet(planet, state, comet_planet_ids)
        if target is None:
            continue

        angle = math.atan2(target.y - planet.y, target.x - planet.x)
        moves.append([planet.id, angle, planet.ships])
        evacuated.add(planet.id)

    return moves, evacuated


def _get_comets_data(obs: Any) -> Any | None:
    """Extract comet group data from observation."""
    if hasattr(obs, "comets"):
        return getattr(obs, "comets", None)
    if isinstance(obs, dict):
        return obs.get("comets")
    return None


def _comet_near_exit(planet_id: int, comets_data: Any) -> bool:
    """Check if a comet planet is within 10 steps of exiting the board."""
    if comets_data is None:
        return False

    groups = comets_data if isinstance(comets_data, list) else []
    if hasattr(comets_data, "__iter__") and not isinstance(comets_data, (str, bytes)):
        groups = list(comets_data)

    for group in groups:
        planets = _get_field(group, "planets", [])
        if not planets:
            continue
        # Check if this planet_id is in this comet group
        pids = []
        for p in (planets if isinstance(planets, list) else list(planets)):
            if hasattr(p, "id"):
                pids.append(int(p.id))
            elif isinstance(p, dict):
                pids.append(int(p.get("id", -1)))
            elif isinstance(p, (int, float)):
                pids.append(int(p))
        if planet_id not in pids:
            continue

        paths = _get_field(group, "paths", None)
        path_index = _get_field(group, "path_index", None)
        if paths is None or path_index is None:
            return False

        # Find path for this planet's quadrant
        idx_in_group = pids.index(planet_id)
        if idx_in_group >= len(paths):
            continue
        path = paths[idx_in_group]
        if not path:
            continue

        # Check remaining path length
        remaining = len(path) - int(path_index)
        return remaining <= 10

    return False


def _get_field(obj: Any, key: str, default: Any) -> Any:
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def _nearest_owned_planet(
    src: PlanetState,
    state: GameState,
    exclude_ids: list[int],
) -> PlanetState | None:
    """Find nearest owned non-comet planet."""
    best: PlanetState | None = None
    best_dist = float("inf")
    for p in state.planets:
        if p.owner != state.player or p.id == src.id or p.id in exclude_ids:
            continue
        d = math.hypot(p.x - src.x, p.y - src.y)
        if d < best_dist:
            best_dist = d
            best = p
    return best
