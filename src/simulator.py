"""Lightweight game simulator for forward projection.

Used by the ExIt search to evaluate candidate actions by simulating
the game state forward N steps without running the full Kaggle environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .features import fleet_hits_planet, fleet_speed, passes_through_sun
from .game_types import FleetState, GameState, PlanetState

MAX_STEPS = 500


@dataclass
class SimState:
    """Lightweight simulation state for forward projection."""

    planet_owner: dict[int, int]  # {pid: owner} (-1 = neutral)
    planet_ships: dict[int, float]  # {pid: ships}
    planet_prod: dict[int, int]  # {pid: production} (shared, not copied)
    # In-flight fleets as scheduled arrivals. Phase 2 (positional simulator):
    # each event carries the straight-line geometry of the fleet so a search-leaf
    # position can be reconstructed in-distribution for the neural value head:
    #   (arrival_step, target_id, owner, ships, launch_step, sx, sy, tx, ty)
    # launch_step == -1 is the sentinel for "no geometry" (positionless legacy
    # producers). Combat/scoring only ever read the first four fields, so the
    # heuristic search path is bit-identical to the pre-geometry simulator.
    fleet_events: list[tuple[int, int, int, int, int, float, float, float, float]]
    current_step: int
    planet_ids: list[int]  # all planet ids (shared, not copied)
    # Phase 1 (every-step in-sim rollout opponent): geometry, precomputed once at
    # the search root and shared immutably across all candidate copies. Both are
    # None on a positionless SimState — the geometry-free rollout policy is only
    # available when these are populated (build_sim_state fills them).
    planet_xy: dict[int, tuple[float, float]] | None = None  # root positions
    neighbors: dict[int, list[int]] | None = None  # pid -> nearest pids

    def copy(self) -> SimState:
        return SimState(
            planet_owner=dict(self.planet_owner),
            planet_ships=dict(self.planet_ships),
            planet_prod=self.planet_prod,  # shared immutable
            fleet_events=list(self.fleet_events),
            current_step=self.current_step,
            planet_ids=self.planet_ids,  # shared immutable
            planet_xy=self.planet_xy,  # shared immutable
            neighbors=self.neighbors,  # shared immutable
        )


def build_sim_state(
    game_state: GameState,
    step: int | None = None,
    with_geometry: bool = False,
) -> SimState:
    """Construct SimState from a parsed GameState observation.

    With `with_geometry`, also precompute root planet positions and per-planet
    nearest-neighbour lists (shared immutably across copies) so the every-step
    rollout opponent (Phase 1) can pick targets and travel times without
    re-running geometry each step. Orbiting planets use their *root* position
    (an approximation the existing search already makes).
    """
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

    planet_xy: dict[int, tuple[float, float]] | None = None
    neighbors: dict[int, list[int]] | None = None
    if with_geometry:
        planet_xy = {p.id: (p.x, p.y) for p in game_state.planets}
        neighbors = {}
        for pid, (x, y) in planet_xy.items():
            others = [
                (math.hypot(ox - x, oy - y), oid)
                for oid, (ox, oy) in planet_xy.items()
                if oid != pid
            ]
            others.sort()
            neighbors[pid] = [oid for _, oid in others]

    return SimState(
        planet_owner=planet_owner,
        planet_ships=planet_ships,
        planet_prod=planet_prod,
        fleet_events=fleet_events,
        current_step=step if step is not None else game_state.step,
        planet_ids=planet_ids,
        planet_xy=planet_xy,
        neighbors=neighbors,
    )


def _build_fleet_schedule(
    fleets: list[FleetState],
    planets: list[PlanetState],
    step: int,
) -> list[tuple[int, int, int, int, int, float, float, float, float]]:
    """Convert existing fleets to scheduled-arrival events with geometry:
    (arrival_step, target_id, owner, ships, launch_step, sx, sy, tx, ty).

    The fleet's *current* position (f.x, f.y) is the segment start and the target
    planet's position is the end, with launch_step = `step` — so linear
    interpolation over [step, arrival] reproduces its straight-line motion (the
    same discrete-arrival approximation the sim already makes)."""
    schedule: list[tuple[int, int, int, int, int, float, float, float, float]] = []
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
            schedule.append(
                (
                    arrival,
                    best_planet.id,
                    f.owner,
                    int(f.ships),
                    step,
                    f.x,
                    f.y,
                    best_planet.x,
                    best_planet.y,
                )
            )
    return schedule


# Rollout-opponent tuning (cheap greedy capture policy; calibrated loosely on
# rule-of-thumb: launch when affordable, keep a small home reserve, favour cheap high-prod
# captures). Deliberately simple — Phase 1's value is that the opponent acts at
# EVERY step, not that it is expert-perfect.
ROLLOUT_MAX_TARGETS = 6  # nearest planets considered per source planet
ROLLOUT_MIN_SHIPS = 3  # don't launch from near-empty planets
ROLLOUT_RESERVE = 1  # ships left behind after a launch


def rollout_launches(
    state: SimState,
    player: int,
    max_targets: int = ROLLOUT_MAX_TARGETS,
) -> list[tuple[int, int, int, float]]:
    """Cheap geometry-light launches for `player` from the current SimState.

    For each owned planet, scan its nearest `max_targets` non-owned planets, score
    each by (production / required_ships) where required accounts for garrison
    growth over the flight, and launch at the single best affordable target.
    Returns [(from_id, target_id, ships, travel_time)]. Requires geometry
    (build_sim_state(..., with_geometry=True)); returns [] otherwise.
    """
    xy = state.planet_xy
    nbrs = state.neighbors
    if xy is None or nbrs is None:
        return []
    out: list[tuple[int, int, int, float]] = []
    for pid in state.planet_ids:
        if state.planet_owner[pid] != player:
            continue
        ships = state.planet_ships[pid]
        if ships < ROLLOUT_MIN_SHIPS:
            continue
        available = int(ships) - ROLLOUT_RESERVE
        if available <= 0:
            continue
        sx, sy = xy[pid]
        best_tid = -1
        best_need = 0
        best_tt = 0.0
        best_score = 0.0
        for tid in nbrs[pid][:max_targets]:
            if state.planet_owner[tid] == player:
                continue
            tx, ty = xy[tid]
            if passes_through_sun(sx, sy, tx, ty):
                continue
            tt = travel_time(sx, sy, tx, ty, available)
            garrison = state.planet_ships[tid]
            # owned-by-enemy planets keep producing during the flight
            grow = state.planet_prod[tid] * tt if state.planet_owner[tid] >= 0 else 0.0
            need = int(garrison + grow) + 1
            if need > available:
                continue
            score = state.planet_prod[tid] / float(need)
            if score > best_score:
                best_score = score
                best_tid, best_need, best_tt = tid, need, tt
        if best_tid >= 0:
            out.append((pid, best_tid, best_need, best_tt))
    return out


def sim_step(
    state: SimState,
    rollout_players: list[int] | None = None,
    launch_fn=None,
) -> None:
    """Advance simulation by 1 step: production, then fleet arrivals + combat.

    When `rollout_players` is given, those players first launch fleets (engine
    order: launch -> production -> move -> combat) — the every-step in-sim
    opponent that stops the lookahead from overrating aggression. By default
    those launches come from the cheap `rollout_launches` heuristic (Phase 1,
    CONFIRMED NEGATIVE — a weak opponent biases the policy to passivity). When
    `launch_fn(state, player) -> [(from_id, target_id, ships, travel_time), ...]`
    is supplied it REPLACES that heuristic: Build 2 passes a closure that rolls
    out the current STRONG distilled net (the AlphaZero-family fix). Both
    arguments None reproduce the original passive-opponent behaviour exactly.
    """
    if rollout_players:
        xy = state.planet_xy
        for pl in rollout_players:
            launches = (
                launch_fn(state, pl) if launch_fn is not None else rollout_launches(state, pl)
            )
            for from_id, tid, ships, tt in launches:
                src_xy = xy[from_id] if xy is not None else None
                dst_xy = xy[tid] if xy is not None else None
                add_fleet_event(state, from_id, tid, ships, tt, src_xy, dst_xy)

    state.current_step += 1

    # Production for all owned planets
    for pid in state.planet_ids:
        if state.planet_owner[pid] >= 0:
            state.planet_ships[pid] += state.planet_prod[pid]

    # Collect arriving fleets this step
    arrivals: dict[int, dict[int, int]] = {}  # {target_id: {owner: total_ships}}
    remaining_events: list[tuple[int, int, int, int, int, float, float, float, float]] = []
    for event in state.fleet_events:
        arr_step, target_id, owner, ships = event[0], event[1], event[2], event[3]
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
    src_xy: tuple[float, float] | None = None,
    dst_xy: tuple[float, float] | None = None,
) -> None:
    """Schedule a fleet arrival by deducting ships and adding an event.

    When `src_xy` and `dst_xy` are given (the aim point used to compute
    travel_time), the event carries the straight-line geometry so the fleet can
    be reconstructed in-distribution at a search leaf (Phase 2). Without them the
    event is positionless (launch_step = -1), matching the legacy behaviour."""
    state.planet_ships[src_id] = max(0, state.planet_ships[src_id] - ships)
    arrival = state.current_step + max(1, int(math.ceil(travel_time)))
    if src_xy is not None and dst_xy is not None:
        ls, (sx, sy), (tx, ty) = state.current_step, src_xy, dst_xy
    else:
        ls, sx, sy, tx, ty = -1, 0.0, 0.0, 0.0, 0.0
    state.fleet_events.append(
        (arrival, target_id, state.planet_owner[src_id], ships, ls, sx, sy, tx, ty)
    )


def fleet_position_at(
    event: tuple[int, int, int, int, int, float, float, float, float],
    step: int,
) -> tuple[float, float, float] | None:
    """(x, y, angle) of an in-flight fleet event at `step`, or None if the event
    carries no geometry (launch_step == -1). Fleets travel a straight line at
    constant speed, so position is a linear interpolation along the segment —
    used to rebuild in-distribution leaf GameStates for the neural value head."""
    arr, _tid, _own, _sh, ls, sx, sy, tx, ty = event
    if ls < 0:
        return None
    dur = max(1, arr - ls)
    frac = (step - ls) / dur
    frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)
    return sx + frac * (tx - sx), sy + frac * (ty - sy), math.atan2(ty - sy, tx - sx)


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
        owner, ships = event[2], event[3]
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
