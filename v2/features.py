"""Feature encoding for V2 pipeline: 40×22 planet matrix + 8 global features."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.features import BOARD_SIZE, SUN_X, SUN_Y, fleet_speed, passes_through_sun
from src.game_types import GameState, PlanetState

from .config import V2EnvConfig
from .state import IncomingFleetInfo, compute_incoming_fleets

PLANET_FEAT_DIM = 22
GLOBAL_FEAT_DIM = 8


@dataclass
class V2Features:
    planet_features: np.ndarray    # [max_planets, PLANET_FEAT_DIM]
    global_features: np.ndarray    # [GLOBAL_FEAT_DIM]
    planet_mask: np.ndarray        # [max_planets] bool — planet exists
    own_mask: np.ndarray           # [max_planets] bool — we own it
    reachability_mask: np.ndarray  # [max_planets, max_planets] bool — can fleet from i reach j
    planet_ids: list[int]          # planet_id per slot (-1 if empty)
    planet_states: list[PlanetState | None]  # state per slot


def encode_features(
    state: GameState,
    cfg: V2EnvConfig,
    comet_ids: list[int] | None = None,
) -> V2Features:
    """Encode full game state into V2 feature tensors."""
    P = cfg.max_planets
    _comet_set: set[int] = set(comet_ids) if comet_ids else set()
    player = state.player

    # Compute incoming fleets
    incoming = compute_incoming_fleets(state, player)

    # Build relative enemy ID mapping
    enemy_ids: list[int] = []
    for p in state.planets:
        if p.owner >= 0 and p.owner != player and p.owner not in enemy_ids:
            enemy_ids.append(p.owner)
    for f in state.fleets:
        if f.owner >= 0 and f.owner != player and f.owner not in enemy_ids:
            enemy_ids.append(f.owner)

    # Planet features [P, 22]
    planet_features = np.zeros((P, PLANET_FEAT_DIM), dtype=np.float32)
    planet_mask = np.zeros(P, dtype=bool)
    own_mask = np.zeros(P, dtype=bool)
    planet_ids: list[int] = [-1] * P
    planet_states: list[PlanetState | None] = [None] * P

    for planet in state.planets:
        slot = planet.id
        if slot < 0 or slot >= P:
            continue

        planet_ids[slot] = planet.id
        planet_states[slot] = planet
        planet_mask[slot] = True
        own_mask[slot] = planet.owner == player

        # Ownership one-hot [mine, enemy1, enemy2, enemy3]
        own = [0.0, 0.0, 0.0, 0.0]
        if planet.owner == player:
            own[0] = 1.0
        elif planet.owner >= 0:
            idx = enemy_ids.index(planet.owner) if planet.owner in enemy_ids else 0
            own[min(idx + 1, 3)] = 1.0

        # Position features
        dx = planet.x - SUN_X
        dy = planet.y - SUN_Y
        dist_center = math.hypot(dx, dy)

        # Incoming fleet info
        info = incoming.get(planet.id, IncomingFleetInfo())

        planet_features[slot] = [
            1.0,                                                    # 0: exists
            1.0 if planet.is_orbiting else 0.0,                     # 1: orbiting
            own[0], own[1], own[2], own[3],                         # 2-5: ownership one-hot
            math.log1p(planet.ships) / 7.0,                         # 6: ships
            planet.x / cfg.board_size,                              # 7: x
            planet.y / cfg.board_size,                              # 8: y
            dist_center / 70.7,                                     # 9: distance from center
            math.sin(math.atan2(dy, dx)) if dist_center > 0.1 else 0.0,  # 10: sin(theta)
            math.cos(math.atan2(dy, dx)) if dist_center > 0.1 else 0.0,  # 11: cos(theta)
            planet.production / 5.0,                                # 12: production
            planet.radius / 4.0,                                    # 13: radius
            math.log1p(info.ships[0]) / 7.0,                       # 14: incoming own ships
            math.log1p(info.ships[1]) / 7.0,                       # 15: incoming enemy1 ships
            math.log1p(info.ships[2]) / 7.0,                       # 16: incoming enemy2 ships
            math.log1p(info.ships[3]) / 7.0,                       # 17: incoming enemy3 ships
            info.eta[0] / 100.0 if info.ships[0] > 0 else 0.0,     # 18: own fleet ETA
            info.eta[1] / 100.0 if info.ships[1] > 0 else 0.0,     # 19: enemy1 fleet ETA
            info.eta[2] / 100.0 if info.ships[2] > 0 else 0.0,     # 20: enemy2 fleet ETA
            info.eta[3] / 100.0 if info.ships[3] > 0 else 0.0,     # 21: enemy3 fleet ETA
        ]

    # Reachability mask [P, P]: True if sending from i to j is a valid action.
    # Combines: (1) sun avoidance, (2) takeover viability, (3) arrival time.
    # Own planets (reinforcement) bypass viability — always valid if reachable.
    reachability_mask = np.zeros((P, P), dtype=bool)
    steps_remaining = max(0, 498 - state.step)

    for i in range(P):
        src = planet_states[i]
        if src is None or src.owner != player:
            continue
        if src.ships <= 0:
            continue

        speed = fleet_speed(src.ships)

        for j in range(P):
            if i == j:
                continue
            tgt = planet_states[j]
            if tgt is None:
                continue

            # Skip comets as targets (unpredictable elliptical orbits)
            if tgt.id in _comet_set:
                continue

            # (1) Sun check
            if passes_through_sun(src.x, src.y, tgt.x, tgt.y):
                continue

            # (2) Arrival time: fleet must arrive before game ends
            dist = math.hypot(src.x - tgt.x, src.y - tgt.y)
            eta = dist / speed if speed > 0 else 999.0
            if eta > steps_remaining:
                continue

            # Own planets: reinforcement is always valid
            if tgt.owner == player:
                reachability_mask[i, j] = True
                continue

            # (3) Takeover viability for enemy/neutral targets
            # Garrison growth: enemy planets produce, neutrals don't
            prod_growth = tgt.production * math.ceil(eta) if tgt.owner >= 0 else 0.0

            # Account for friendly fleets already en route to this target
            tgt_info = incoming.get(tgt.id, IncomingFleetInfo())
            friendly_incoming = tgt_info.ships[0]  # own team's incoming ships

            # Must have enough that a reasonable allocation (50%) can capture.
            # This prevents the model from considering targets it can only
            # capture by sending 100% of ships (too restrictive in practice).
            effective_garrison = tgt.ships + prod_growth - friendly_incoming
            if src.ships >= 2 * (effective_garrison + 1):
                reachability_mask[i, j] = True

    # Global features [8]
    my_ships = 0.0
    my_prod = 0.0
    best_enemy_ships = 0.0
    best_enemy_prod = 0.0
    my_planets = 0
    total_planets = 0
    enemy_ship_totals: dict[int, float] = {}
    enemy_prod_totals: dict[int, float] = {}

    for p in state.planets:
        total_planets += 1
        if p.owner == player:
            my_ships += p.ships
            my_prod += p.production
            my_planets += 1
        elif p.owner >= 0:
            enemy_ship_totals[p.owner] = enemy_ship_totals.get(p.owner, 0.0) + p.ships
            enemy_prod_totals[p.owner] = enemy_prod_totals.get(p.owner, 0.0) + p.production

    for f in state.fleets:
        if f.owner == player:
            my_ships += f.ships
        elif f.owner >= 0:
            enemy_ship_totals[f.owner] = enemy_ship_totals.get(f.owner, 0.0) + f.ships

    if enemy_ship_totals:
        best_enemy_ships = max(enemy_ship_totals.values())
    if enemy_prod_totals:
        best_enemy_prod = max(enemy_prod_totals.values())

    global_features = np.array([
        state.step / 500.0,                                          # 0: step
        state.angular_velocity / 0.05,                               # 1: angular velocity
        math.log1p(my_ships) / 10.0,                                 # 2: own ships (log)
        math.log1p(best_enemy_ships) / 10.0,                         # 3: best enemy ships (log)
        my_prod / max(my_prod + best_enemy_prod, 1.0),               # 4: own prod fraction
        best_enemy_prod / max(my_prod + best_enemy_prod, 1.0),       # 5: enemy prod fraction
        my_planets / max(cfg.max_planets, 1),                        # 6: own planets fraction
        total_planets / max(cfg.max_planets, 1),                     # 7: total planets fraction
    ], dtype=np.float32)

    return V2Features(
        planet_features=planet_features,
        global_features=global_features,
        planet_mask=planet_mask,
        own_mask=own_mask,
        reachability_mask=reachability_mask,
        planet_ids=planet_ids,
        planet_states=planet_states,
    )
