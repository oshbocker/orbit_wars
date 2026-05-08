from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

SUN_X, SUN_Y = 50.0, 50.0


@dataclass(slots=True)
class PlanetState:
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: int
    production: int
    is_orbiting: bool = False
    orbital_radius: float = 0.0
    initial_angle: float = 0.0


@dataclass(slots=True)
class FleetState:
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: int


@dataclass
class GameState:
    step: int
    player: int
    planets: list[PlanetState]
    fleets: list[FleetState]
    angular_velocity: float = 0.0
    planets_by_id: dict[int, PlanetState] = field(default_factory=dict, repr=False)


def _obs_get(obs: Any, key: str, default: Any = None) -> Any:
    if hasattr(obs, key):
        return getattr(obs, key)
    if isinstance(obs, dict):
        return obs.get(key, default)
    return default


def _parse_planet(p: Any) -> PlanetState:
    if hasattr(p, "production"):
        return PlanetState(
            id=int(p.id), owner=int(p.owner),
            x=float(p.x), y=float(p.y),
            radius=float(p.radius), ships=int(p.ships),
            production=int(p.production),
        )
    if isinstance(p, dict):
        return PlanetState(
            id=int(p["id"]), owner=int(p["owner"]),
            x=float(p["x"]), y=float(p["y"]),
            radius=float(p["radius"]), ships=int(p["ships"]),
            production=int(p["production"]),
        )
    return PlanetState(
        id=int(p[0]), owner=int(p[1]),
        x=float(p[2]), y=float(p[3]),
        radius=float(p[4]), ships=int(p[5]),
        production=int(p[6]),
    )


def _parse_fleet(f: Any) -> FleetState:
    if hasattr(f, "from_planet_id"):
        return FleetState(
            id=int(f.id), owner=int(f.owner),
            x=float(f.x), y=float(f.y),
            angle=float(f.angle), from_planet_id=int(f.from_planet_id),
            ships=int(f.ships),
        )
    if isinstance(f, dict):
        return FleetState(
            id=int(f["id"]), owner=int(f["owner"]),
            x=float(f["x"]), y=float(f["y"]),
            angle=float(f["angle"]), from_planet_id=int(f["from_planet_id"]),
            ships=int(f["ships"]),
        )
    return FleetState(
        id=int(f[0]), owner=int(f[1]),
        x=float(f[2]), y=float(f[3]),
        angle=float(f[4]), from_planet_id=int(f[5]),
        ships=int(f[6]),
    )


def _detect_orbiting(
    planets: list[PlanetState],
    raw_initial: list[Any],
    angular_velocity: float,
) -> None:
    if not raw_initial:
        return

    init_by_id: dict[int, tuple[float, float]] = {}
    for ip in raw_initial:
        if hasattr(ip, "id"):
            init_by_id[int(ip.id)] = (float(ip.x), float(ip.y))
        elif isinstance(ip, dict):
            init_by_id[int(ip["id"])] = (float(ip["x"]), float(ip["y"]))
        else:
            init_by_id[int(ip[0])] = (float(ip[2]), float(ip[3]))

    for p in planets:
        if p.id not in init_by_id:
            continue
        ix, iy = init_by_id[p.id]
        orb_r = math.hypot(ix - SUN_X, iy - SUN_Y)
        if orb_r + p.radius < 50.0 and orb_r > 0.1:
            p.is_orbiting = True
            p.orbital_radius = orb_r
            p.initial_angle = math.atan2(iy - SUN_Y, ix - SUN_X)


def parse_observation(obs: Any) -> GameState:
    player = int(_obs_get(obs, "player", 0))
    step = int(_obs_get(obs, "step", 0))
    angular_velocity = float(_obs_get(obs, "angular_velocity", 0.0))
    raw_initial = _obs_get(obs, "initial_planets", []) or []

    planets = [_parse_planet(p) for p in (_obs_get(obs, "planets", []) or [])]
    fleets = [_parse_fleet(f) for f in (_obs_get(obs, "fleets", []) or [])]

    _detect_orbiting(planets, raw_initial, angular_velocity)

    planets_by_id = {p.id: p for p in planets}
    return GameState(
        step=step,
        player=player,
        planets=planets,
        fleets=fleets,
        angular_velocity=angular_velocity,
        planets_by_id=planets_by_id,
    )
