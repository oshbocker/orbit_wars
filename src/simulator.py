"""Lightweight game simulator for forward projection.

Used by the ExIt search to evaluate candidate actions by simulating
the game state forward N steps without running the full Kaggle environment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .features import fleet_speed, passes_through_sun, fleet_hits_planet, BOARD_SIZE, MAX_SHIP_SPEED
from .game_types import GameState, PlanetState, FleetState, SUN_X, SUN_Y

MAX_STEPS = 500


@dataclass
class SimState:
    """Lightweight simulation state for forward projection."""
    planet_owner: dict[int, int]       # {pid: owner} (-1 = neutral)
    planet_ships: dict[int, float]     # {pid: ships}
    planet_prod: dict[int, int]        # {pid: production} (shared, not copied)
    fleet_events: list[tuple[int, int, int, int]]  # [(arrival_step, target_id, owner, ships)]
    current_step: int
    planet_ids: list[int]              # all planet ids (shared, not copied)

    def copy(self) -> SimState:
        return SimState(
            planet_owner=dict(self.planet_owner),
            planet_ships=dict(self.planet_ships),
            planet_prod=self.planet_prod,       # shared immutable
            fleet_events=list(self.fleet_events),
            current_step=self.current_step,
            planet_ids=self.planet_ids,          # shared immutable
        )


def build_sim_state(game_state: GameState, step: int | None = None) -> SimState:
    """Construct SimState from a parsed GameState observation."""
    planet_owner: dict[int, int] = {}
    planet_ships: dict[int, float] = {}
    planet_prod: dict[int, int] = {}
    planet_ids: list[int] = []

    for p in game_state.planets:
        planet_owner[p.id] = p.owner
        planet_ships[p.id] = float(p.ships)
        planet_prod[p.id] = p.production
        planet_ids.append(p.id)

    # Convert existing fleets to scheduled arrival events
    fleet_events = _build_fleet_schedule(game_state.fleets, game_state.planets, game_state.step)

    return SimState(
        planet_owner=planet_owner,
        planet_ships=planet_ships,
        planet_prod=planet_prod,
        fleet_events=fleet_events,
        current_step=step if step is not None else game_state.step,
        planet_ids=planet_ids,
    )


def _build_fleet_schedule(
    fleets: list[FleetState],
    planets: list[PlanetState],
    step: int,
) -> list[tuple[int, int, int, int]]:
    """Convert existing fleets to (arrival_step, target_id, owner, ships) tuples."""
    schedule: list[tuple[int, int, int, int]] = []
    for f in fleets:
        best_planet: PlanetState | None = None
        best_eta = float("inf")
        for p in planets:
            eta = fleet_hits_planet(f, p)
            if eta is not None and eta < best_eta:
                best_eta = eta
                best_planet = p
        if best_planet is not None:
            arrival = step + max(1, int(math.ceil(best_eta)))
            schedule.append((arrival, best_planet.id, f.owner, int(f.ships)))
    return schedule


def sim_step(state: SimState) -> None:
    """Advance simulation by 1 step: production, then fleet arrivals + combat."""
    state.current_step += 1

    # Production for all owned planets
    for pid in state.planet_ids:
        if state.planet_owner[pid] >= 0:
            state.planet_ships[pid] += state.planet_prod[pid]

    # Collect arriving fleets this step
    arrivals: dict[int, dict[int, int]] = {}  # {target_id: {owner: total_ships}}
    remaining_events: list[tuple[int, int, int, int]] = []
    for event in state.fleet_events:
        arr_step, target_id, owner, ships = event
        if arr_step <= state.current_step:
            if target_id not in arrivals:
                arrivals[target_id] = {}
            arrivals[target_id][owner] = arrivals[target_id].get(owner, 0) + ships
        else:
            remaining_events.append(event)
    state.fleet_events = remaining_events

    # Combat resolution per planet
    for target_id, attackers in arrivals.items():
        if target_id not in state.planet_owner:
            continue
        defender = state.planet_owner[target_id]
        garrison = state.planet_ships[target_id]

        # Add garrison to defender's total
        if defender >= 0:
            attackers[defender] = attackers.get(defender, 0) + int(garrison)
        else:
            attackers[-1] = attackers.get(-1, 0) + int(garrison)

        # Find top two
        sorted_forces = sorted(attackers.items(), key=lambda x: -x[1])
        if not sorted_forces:
            continue

        top_owner, top_ships = sorted_forces[0]
        second_ships = sorted_forces[1][1] if len(sorted_forces) > 1 else 0

        survivors = top_ships - second_ships
        if survivors <= 0:
            state.planet_owner[target_id] = -1
            state.planet_ships[target_id] = 0
        elif top_owner == defender:
            state.planet_ships[target_id] = survivors
        else:
            state.planet_owner[target_id] = top_owner
            state.planet_ships[target_id] = survivors


def add_fleet_event(
    state: SimState,
    src_id: int,
    target_id: int,
    ships: int,
    travel_time: float,
) -> None:
    """Schedule a fleet arrival by deducting ships and adding an event."""
    state.planet_ships[src_id] = max(0, state.planet_ships[src_id] - ships)
    arrival = state.current_step + max(1, int(math.ceil(travel_time)))
    state.fleet_events.append((arrival, target_id, state.planet_owner[src_id], ships))


def evaluate_state(state: SimState, player: int) -> float:
    """Score a simulation state for a player.

    Score = (my_ships - best_enemy_ships) + prod_weight * (my_prod - best_enemy_prod)
    Designed to reward gaining advantage over enemies.
    """
    my_ships = 0.0
    my_prod = 0.0
    enemy_ships: dict[int, float] = {}
    enemy_prod: dict[int, float] = {}

    for pid in state.planet_ids:
        owner = state.planet_owner[pid]
        ships = state.planet_ships[pid]
        prod = state.planet_prod[pid]
        if owner == player:
            my_ships += ships
            my_prod += prod
        elif owner >= 0:
            enemy_ships[owner] = enemy_ships.get(owner, 0.0) + ships
            enemy_prod[owner] = enemy_prod.get(owner, 0.0) + prod

    # Count fleet ships
    for event in state.fleet_events:
        _, _, owner, ships = event
        if owner == player:
            my_ships += ships
        elif owner >= 0:
            enemy_ships[owner] = enemy_ships.get(owner, 0.0) + ships

    best_enemy_ships = max(enemy_ships.values(), default=0.0)
    best_enemy_prod = max(enemy_prod.values(), default=0.0)

    remaining = max(0, MAX_STEPS - state.current_step)
    prod_weight = min(15.0, remaining / 8.0)

    ship_advantage = my_ships - best_enemy_ships
    prod_advantage = my_prod - best_enemy_prod

    return ship_advantage + prod_weight * prod_advantage


def travel_time(src_x: float, src_y: float, dst_x: float, dst_y: float, ships: int) -> float:
    """Compute travel time between two points for a fleet of given size."""
    d = math.hypot(dst_x - src_x, dst_y - src_y)
    speed = fleet_speed(ships)
    return d / max(speed, 0.1)
