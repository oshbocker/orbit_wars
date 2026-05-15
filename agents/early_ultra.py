"""
Early Ultra agent for Orbit Wars.

Imports everything from ultra.py but overrides the opening logic (turns 0-59)
with a greedy production-maximizing planner. After turn 60, falls back to
ultra's _plan_moves() unchanged.

Key differences from ultra's opening:
- Greedy forward-simulation instead of beam search
- Models compound production (captured planets become sources)
- Scores by prod * (60 - capture_turn) instead of prod^1.3 * remaining_turns
- Tight margins (1 ship only, no static bonus or long-trip extras)
- Falls back to ultra's _opening_expand for defense/rescue when greedy has no moves
- Greedy phase limited to first 30 steps; ultra's full opening takes over after
"""

import math
import time
from collections import defaultdict

from agents.ultra import (
    World, _plan_moves, _opening_expand, _evacuate_comets,
    _read, _parse_planet, _parse_fleet,
    _is_static, _dist, _fleet_speed,
    MAX_SPEED, SOFT_DEADLINE_FRAC,
)


def _greedy_opening(world, deadline):
    """Greedy forward-simulation planner for turns 0-59.

    Maximizes total production-turns within the opening window by greedily
    selecting the highest-scoring (source, target) pair. Captured planets
    become new sources for future captures (compound production).

    Returns moves for the current turn, or empty list if no turn-0 launches.
    """
    window = max(0, 60 - world.step)  # planning horizon extends to step 60
    if window <= 0:
        return []

    # Source state: pid -> [ships_available, production, available_at]
    # available_at: turns from now when this source has ships available
    sources = {}
    for p in world.my_planets:
        # Minimal reserves: only hold ships for active incoming threats
        keep = world.timeline[p.id]["keep_needed"]
        available = max(0, int(p.ships) - keep)
        sources[p.id] = [available, int(p.production), 0]

    # Account for in-flight friendly captures (become future sources)
    for pid, arrs in world.arrivals.items():
        p = world.by_id.get(pid)
        if p is None or p.owner == world.player or pid in sources:
            continue
        friendly = sorted(
            [(eta, ships) for eta, owner, ships in arrs if owner == world.player],
            key=lambda x: x[0])
        if not friendly:
            continue
        garrison = int(p.ships)
        cum = 0
        for eta, ships in friendly:
            cum += ships
            if cum > garrison:
                sources[pid] = [cum - garrison, int(p.production),
                                int(math.ceil(eta))]
                break

    committed = set()  # target IDs already assigned
    planned = []       # [(src_id, tgt_id, launch_turn_from_now, ships_to_send)]

    # Greedy loop: find best (source, target) pair, simulate capture, repeat
    for _ in range(20):  # safety limit
        if deadline and time.perf_counter() > deadline:
            break

        best_score = 0
        best_plan = None

        for target in world.planets:
            if deadline and time.perf_counter() > deadline:
                break
            # Skip owned, already committed, or already being captured
            if target.owner == world.player or target.id in committed:
                continue
            if target.id in sources:
                continue

            for src_id, (src_ships, src_prod, src_avail) in list(sources.items()):
                if src_ships <= 0 and src_prod <= 0:
                    continue

                src_planet = world.by_id[src_id]
                d = _dist(src_planet.x, src_planet.y, target.x, target.y)
                if d < 0.1:
                    continue

                # Quick reject: even max-speed fleet can't reach within window
                min_eta = max(1, int(math.ceil(d / MAX_SPEED)))
                if src_avail + min_eta >= window:
                    continue

                # For turn-0 launches from currently owned sources, use
                # world.best_aim for accurate ETA (handles sun, orbits)
                use_aim = (src_avail == 0 and src_ships > 0
                           and src_id in {p.id for p in world.my_planets})
                if use_aim:
                    aim_result = world.best_aim(src_id, target.id,
                                                max(1, src_ships))
                    if aim_result is None:
                        continue
                    _, (_, aim_turns, _, _) = aim_result
                    rough_eta = aim_turns
                else:
                    rough_fleet = max(1, src_ships)
                    rough_speed = _fleet_speed(rough_fleet)
                    rough_eta = max(1, int(math.ceil(d / rough_speed)))

                rough_capture = src_avail + rough_eta
                if rough_capture >= window:
                    continue

                # Ships needed to capture at estimated arrival time
                needed = world.ships_to_own(target.id, rough_capture) + 1
                if needed <= 0:
                    needed = 1

                # Wait for ship accumulation if needed
                wait = 0
                if needed > src_ships:
                    if src_prod <= 0:
                        continue
                    wait = int(math.ceil((needed - src_ships) / src_prod))

                fleet = src_ships + src_prod * wait
                launch_turn = src_avail + wait

                # Refine ETA with actual fleet size (more ships = faster)
                speed = _fleet_speed(max(1, fleet))
                eta = max(1, int(math.ceil(d / speed)))
                capture_turn = launch_turn + eta

                if capture_turn >= window:
                    continue

                # Re-check needed at refined capture turn
                needed2 = world.ships_to_own(target.id, capture_turn) + 1
                if needed2 <= 0:
                    needed2 = 1
                if needed2 > fleet:
                    # Need more ships — one more refinement
                    if src_prod <= 0:
                        continue
                    extra = int(math.ceil((needed2 - fleet) / src_prod))
                    wait += extra
                    fleet = src_ships + src_prod * wait
                    launch_turn = src_avail + wait
                    speed = _fleet_speed(max(1, fleet))
                    eta = max(1, int(math.ceil(d / speed)))
                    capture_turn = launch_turn + eta
                    if capture_turn >= window:
                        continue
                    needed3 = world.ships_to_own(target.id, capture_turn) + 1
                    if needed3 > fleet:
                        continue
                    needed = needed3
                else:
                    needed = needed2

                # Score: total production-turns gained within opening window
                score = target.production * (window - capture_turn)
                if score <= 0:
                    continue

                # Safety multipliers based on reaction times
                my_react = world.my_react.get(target.id, 1e9)
                en_react = world.enemy_react.get(target.id, 1e9)
                if en_react < capture_turn:
                    score *= 0.5  # enemy arrives first
                elif abs(my_react - en_react) <= 2:
                    score *= 0.7  # contested

                # Static target bonus (easier to aim at reliably)
                if _is_static(target, world.initial_by_id):
                    score *= 1.1

                if score > best_score:
                    best_score = score
                    best_plan = (src_id, target.id, launch_turn,
                                 int(needed), wait, capture_turn)

        if best_plan is None:
            break

        # Simulate capture
        src_id, tgt_id, launch_turn, ships_sent, wait, capture_turn = best_plan
        committed.add(tgt_id)

        # Update source: deduct ships sent
        src_ships, src_prod, src_avail = sources[src_id]
        ships_at_launch = src_ships + src_prod * wait
        remaining = max(0, ships_at_launch - ships_sent)
        sources[src_id] = [remaining, src_prod, launch_turn]

        # Add captured target as new source (compound production)
        target = world.by_id[tgt_id]
        residual = max(0, ships_sent - int(target.ships))
        sources[tgt_id] = [residual, int(target.production), capture_turn]

        planned.append((src_id, tgt_id, launch_turn, ships_sent))

    # Execute current-turn launches only
    moves = []
    sent_from = defaultdict(int)

    for src_id, tgt_id, launch_turn, ships in planned:
        if launch_turn > 0:
            continue  # future launch, skip this turn

        src = world.by_id[src_id]
        available = max(0, int(src.ships) - sent_from[src_id])
        actual = min(ships, available)
        if actual < 1:
            continue

        shot = world.aim(src_id, tgt_id, actual)
        if shot is None:
            # Try with best_aim as fallback (different ship count may work)
            result = world.best_aim(src_id, tgt_id, actual)
            if result is None:
                continue
            aim_ships, (angle, turns, tx, ty) = result
            actual = min(actual, aim_ships)
            if actual < 1:
                continue
            shot = (angle, turns, tx, ty)

        moves.append([src_id, float(shot[0]), actual])
        sent_from[src_id] += actual

    return moves


def agent(obs, config=None):
    """Early Ultra agent: greedy production-maximizing opening + ultra mid/late game."""
    start = time.perf_counter()

    player = _read(obs, "player", 0)
    raw_planets = _read(obs, "planets", []) or []
    raw_fleets = _read(obs, "fleets", []) or []
    step = _read(obs, "step", 0) or 0
    ang_vel = _read(obs, "angular_velocity", 0.0) or 0.0
    raw_init = _read(obs, "initial_planets", []) or []
    comets = _read(obs, "comets", []) or []
    comet_ids = set(_read(obs, "comet_planet_ids", []) or [])

    planets = [_parse_planet(p) for p in raw_planets]
    fleets = [_parse_fleet(f) for f in raw_fleets]
    initial_planets = [_parse_planet(p) for p in raw_init]
    initial_by_id = {p.id: p for p in initial_planets}

    world = World(player, step, planets, fleets, initial_by_id,
                  ang_vel, comets, comet_ids)

    if not world.my_planets:
        return []

    # Time budget
    act_timeout = _read(config, "actTimeout", 1.0) if config else 1.0
    soft = min(SOFT_DEADLINE_FRAC, max(0.55, act_timeout * 0.82))
    deadline = start + soft

    if step < 30:
        # Greedy production-maximizing opening for first 30 steps
        moves = _greedy_opening(world, deadline)
        if not moves:
            # Fall back to ultra's opening expand for defense/rescue/swarm
            moves = _opening_expand(world, deadline)
    elif world.is_opening:
        # Ultra's full opening planner for steps 30-79
        moves = _opening_expand(world, deadline)
    else:
        moves = _plan_moves(world, deadline)

    return _evacuate_comets(world, moves)
