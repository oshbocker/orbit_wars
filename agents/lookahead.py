"""
Lookahead agent for Orbit Wars.

Forward-simulation decision engine that evaluates moves by projecting game state
N steps into the future and picking the action that maximizes total ships. Uses a
greedy sequential planner with several key improvements:
  - Two-pass decision loop: planets can split forces across multiple targets
  - Committed tracking: rollout avoids wasteful double-targeting
  - Enemy decay modeling: enemy garrisons shrink realistically during sim
  - Heuristic pre-filter: only top candidates get full simulation
  - Production-advantage evaluation: weighs production dominance heavily
"""

import math
import time

# ── constants ──────────────────────────────────────────────────────────────
SUN_X, SUN_Y, SUN_RADIUS = 50.0, 50.0, 10.0
SUN_SAFE_RADIUS = SUN_RADIUS + 2.0
MAX_SHIP_SPEED = 6.0
BOARD_SIZE = 100.0
MAX_STEPS = 500

# Lookahead-specific
LOOKAHEAD_DEPTH = 20
MAX_SOURCE_PLANETS = 7
MAX_TARGETS_PER_SOURCE = 10
SHIP_FRACTIONS = (1.0, 0.75, 0.5)
ROLLOUT_NEIGHBORS = 5
ROLLOUT_MAX_LAUNCHES = 3

# ── helpers (reused from advanced.py) ──────────────────────────────────────

def _dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _fleet_speed(ships):
    if ships <= 1:
        return 1.0
    return 1.0 + (MAX_SHIP_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def _travel_time(x1, y1, x2, y2, ships):
    d = _dist(x1, y1, x2, y2)
    speed = _fleet_speed(ships)
    if speed <= 0:
        return 9999.0
    return d / speed


def _passes_through_sun(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - SUN_X, y1 - SUN_Y
    a = dx * dx + dy * dy
    if a == 0:
        return False
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - SUN_SAFE_RADIUS ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 < t2)


def _safe_angle(src_x, src_y, dst_x, dst_y):
    direct = math.atan2(dst_y - src_y, dst_x - src_x)
    if not _passes_through_sun(src_x, src_y, dst_x, dst_y):
        return direct, False

    cx, cy = SUN_X, SUN_Y
    a_src = math.atan2(src_y - cy, src_x - cx)
    r = SUN_SAFE_RADIUS + 3.0
    best_wp = None
    best_total = float("inf")
    for offset in (math.pi / 2, -math.pi / 2):
        wp_a = a_src + offset
        wx = cx + r * math.cos(wp_a)
        wy = cy + r * math.sin(wp_a)
        wx = max(1.0, min(99.0, wx))
        wy = max(1.0, min(99.0, wy))
        total = _dist(src_x, src_y, wx, wy) + _dist(wx, wy, dst_x, dst_y)
        if total < best_total and not _passes_through_sun(src_x, src_y, wx, wy):
            best_total = total
            best_wp = (wx, wy)

    if best_wp is None:
        for offset in (math.pi / 3, -math.pi / 3, 2 * math.pi / 3, -2 * math.pi / 3):
            wp_a = a_src + offset
            wx = cx + r * math.cos(wp_a)
            wy = cy + r * math.sin(wp_a)
            wx = max(1.0, min(99.0, wx))
            wy = max(1.0, min(99.0, wy))
            if not _passes_through_sun(src_x, src_y, wx, wy):
                best_wp = (wx, wy)
                break

    if best_wp is None:
        return direct, False

    angle = math.atan2(best_wp[1] - src_y, best_wp[0] - src_x)
    return angle, True


# ── lightweight data classes ───────────────────────────────────────────────

class _Planet:
    __slots__ = ("id", "owner", "x", "y", "radius", "ships", "production",
                 "is_orbiting", "orbital_radius", "initial_angle", "angular_velocity")

    def __init__(self, id, owner, x, y, radius, ships, production,
                 is_orbiting=False, orbital_radius=0.0, initial_angle=0.0,
                 angular_velocity=0.0):
        self.id = id
        self.owner = owner
        self.x = x
        self.y = y
        self.radius = radius
        self.ships = ships
        self.production = production
        self.is_orbiting = is_orbiting
        self.orbital_radius = orbital_radius
        self.initial_angle = initial_angle
        self.angular_velocity = angular_velocity


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


# ── observation parsing ────────────────────────────────────────────────────

def _get(obs, key, default=None):
    if hasattr(obs, key):
        return getattr(obs, key)
    if isinstance(obs, dict):
        return obs.get(key, default)
    return default


def _parse_planet(p):
    if hasattr(p, "production"):
        return _Planet(p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
    if isinstance(p, dict):
        return _Planet(p["id"], p["owner"], p["x"], p["y"], p["radius"], p["ships"], p["production"])
    return _Planet(p[0], p[1], p[2], p[3], p[4], p[5], p[6])


def _parse_fleet(f):
    if hasattr(f, "from_planet_id"):
        return _Fleet(f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships)
    if isinstance(f, dict):
        return _Fleet(f["id"], f["owner"], f["x"], f["y"], f["angle"], f["from_planet_id"], f["ships"])
    return _Fleet(f[0], f[1], f[2], f[3], f[4], f[5], f[6])


# ── orbit detection & prediction ───────────────────────────────────────────

def _detect_orbiting(planets, initial_planets, angular_velocity):
    if not initial_planets:
        return

    init_by_id = {}
    for ip in initial_planets:
        if hasattr(ip, "id"):
            init_by_id[ip.id] = ip
        elif isinstance(ip, dict):
            init_by_id[ip["id"]] = ip
        else:
            init_by_id[ip[0]] = ip

    for p in planets:
        if p.id not in init_by_id:
            continue
        ip = init_by_id[p.id]
        if hasattr(ip, "x"):
            ix, iy = ip.x, ip.y
        elif isinstance(ip, dict):
            ix, iy = ip["x"], ip["y"]
        else:
            ix, iy = ip[2], ip[3]

        orb_r = _dist(ix, iy, SUN_X, SUN_Y)
        if orb_r + p.radius < 50.0 and orb_r > 0.1:
            p.is_orbiting = True
            p.orbital_radius = orb_r
            p.initial_angle = math.atan2(iy - SUN_Y, ix - SUN_X)
            p.angular_velocity = angular_velocity


def _planet_pos_at(planet, future_step):
    if not planet.is_orbiting:
        return planet.x, planet.y
    angle = planet.initial_angle + planet.angular_velocity * future_step
    x = SUN_X + planet.orbital_radius * math.cos(angle)
    y = SUN_Y + planet.orbital_radius * math.sin(angle)
    return x, y


def _aim_at_moving(src, target, step, ships):
    if not target.is_orbiting:
        angle, _ = _safe_angle(src.x, src.y, target.x, target.y)
        spawn_x = src.x + src.radius * math.cos(angle)
        spawn_y = src.y + src.radius * math.sin(angle)
        tt = _travel_time(spawn_x, spawn_y, target.x, target.y, ships)
        return angle, tt

    speed = _fleet_speed(ships)
    R = target.orbital_radius
    omega = target.angular_velocity

    def _intercept_error(t):
        a = target.initial_angle + omega * (step + t)
        tx = SUN_X + R * math.cos(a)
        ty = SUN_Y + R * math.sin(a)
        theta = math.atan2(ty - src.y, tx - src.x)
        sx = src.x + src.radius * math.cos(theta)
        sy = src.y + src.radius * math.sin(theta)
        d = _dist(sx, sy, tx, ty)
        return d / speed - t

    max_t = 80.0
    dt = 1.0
    prev_t = 0.5
    prev_err = _intercept_error(prev_t)
    bracket = None

    t = prev_t + dt
    while t <= max_t:
        err = _intercept_error(t)
        if prev_err >= 0 and err <= 0:
            bracket = (prev_t, t)
            break
        prev_t = t
        prev_err = err
        t += dt

    if bracket is None:
        angle, _ = _safe_angle(src.x, src.y, target.x, target.y)
        spawn_x = src.x + src.radius * math.cos(angle)
        spawn_y = src.y + src.radius * math.sin(angle)
        tt = _travel_time(spawn_x, spawn_y, target.x, target.y, ships)
        return angle, tt

    lo, hi = bracket
    for _ in range(25):
        mid = (lo + hi) / 2.0
        if _intercept_error(mid) > 0:
            lo = mid
        else:
            hi = mid

    t_intercept = (lo + hi) / 2.0
    a = target.initial_angle + omega * (step + t_intercept)
    tx = SUN_X + R * math.cos(a)
    ty = SUN_Y + R * math.sin(a)

    angle = math.atan2(ty - src.y, tx - src.x)
    spawn_x = src.x + src.radius * math.cos(angle)
    spawn_y = src.y + src.radius * math.sin(angle)

    if not _passes_through_sun(spawn_x, spawn_y, tx, ty):
        tt = _dist(spawn_x, spawn_y, tx, ty) / speed
        return angle, tt

    prev_t = bracket[1]
    prev_err = _intercept_error(prev_t)
    t = prev_t + dt
    while t <= max_t:
        err = _intercept_error(t)
        if prev_err >= 0 and err <= 0:
            lo, hi = prev_t, t
            for _ in range(25):
                mid = (lo + hi) / 2.0
                if _intercept_error(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            t_intercept = (lo + hi) / 2.0
            a = target.initial_angle + omega * (step + t_intercept)
            tx = SUN_X + R * math.cos(a)
            ty = SUN_Y + R * math.sin(a)
            angle = math.atan2(ty - src.y, tx - src.x)
            spawn_x = src.x + src.radius * math.cos(angle)
            spawn_y = src.y + src.radius * math.sin(angle)
            if not _passes_through_sun(spawn_x, spawn_y, tx, ty):
                tt = _dist(spawn_x, spawn_y, tx, ty) / speed
                return angle, tt
        prev_t = t
        prev_err = err
        t += dt

    a = target.initial_angle + omega * (step + (bracket[0] + bracket[1]) / 2.0)
    tx = SUN_X + R * math.cos(a)
    ty = SUN_Y + R * math.sin(a)
    angle, _ = _safe_angle(src.x, src.y, tx, ty)
    spawn_x = src.x + src.radius * math.cos(angle)
    spawn_y = src.y + src.radius * math.sin(angle)
    tt = _dist(spawn_x, spawn_y, tx, ty) / speed
    return angle, tt


# ── fleet-planet collision ─────────────────────────────────────────────────

def _fleet_hits_planet(fleet, planet):
    speed = _fleet_speed(fleet.ships)
    dx = math.cos(fleet.angle) * speed
    dy = math.sin(fleet.angle) * speed

    fx = fleet.x - planet.x
    fy = fleet.y - planet.y
    hit_r = planet.radius + 0.5

    a = dx * dx + dy * dy
    if a < 1e-10:
        return None
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - hit_r * hit_r

    disc = b * b - 4 * a * c
    if disc < 0:
        return None

    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)

    t = t1 if t1 > 0 else t2
    if t <= 0:
        return None

    hit_x = fleet.x + dx * t
    hit_y = fleet.y + dy * t
    if hit_x < 0 or hit_x > BOARD_SIZE or hit_y < 0 or hit_y > BOARD_SIZE:
        d_edge = min(
            (BOARD_SIZE - fleet.x) / max(dx, 1e-10) if dx > 0 else float("inf"),
            -fleet.x / min(dx, -1e-10) if dx < 0 else float("inf"),
            (BOARD_SIZE - fleet.y) / max(dy, 1e-10) if dy > 0 else float("inf"),
            -fleet.y / min(dy, -1e-10) if dy < 0 else float("inf"),
        )
        if t > d_edge:
            return None

    sun_fx = fleet.x - SUN_X
    sun_fy = fleet.y - SUN_Y
    sun_b = 2 * (sun_fx * dx + sun_fy * dy)
    sun_c = sun_fx * sun_fx + sun_fy * sun_fy - SUN_RADIUS * SUN_RADIUS
    sun_disc = sun_b * sun_b - 4 * a * sun_c
    if sun_disc >= 0:
        sun_sq = math.sqrt(sun_disc)
        sun_t = (-sun_b - sun_sq) / (2 * a)
        if 0 < sun_t < t:
            return None

    return t


def _compute_threats(fleets, my_planets, player):
    threats = {}
    enemy_fleets = [f for f in fleets if f.owner != player]
    for p in my_planets:
        for f in enemy_fleets:
            eta = _fleet_hits_planet(f, p)
            if eta is not None and eta < 15:
                threats[p.id] = threats.get(p.id, 0) + f.ships
    return threats


def _estimate_in_flight(fleets, planets, player):
    in_flight = {}
    my_fleets = [f for f in fleets if f.owner == player]
    for f in my_fleets:
        best_planet = None
        best_eta = float("inf")
        for p in planets:
            eta = _fleet_hits_planet(f, p)
            if eta is not None and eta < best_eta:
                best_eta = eta
                best_planet = p
        if best_planet is not None:
            in_flight[best_planet.id] = in_flight.get(best_planet.id, 0) + f.ships
    return in_flight


# ── precomputation ─────────────────────────────────────────────────────────

def _precompute_distances(planets):
    """Compute distance matrix and sun-blocked matrix between all planet pairs."""
    n = len(planets)
    dist_matrix = {}
    sun_blocked = {}
    for i in range(n):
        pi = planets[i]
        for j in range(i + 1, n):
            pj = planets[j]
            d = _dist(pi.x, pi.y, pj.x, pj.y)
            dist_matrix[(pi.id, pj.id)] = d
            dist_matrix[(pj.id, pi.id)] = d
            blocked = _passes_through_sun(pi.x, pi.y, pj.x, pj.y)
            sun_blocked[(pi.id, pj.id)] = blocked
            sun_blocked[(pj.id, pi.id)] = blocked
    return dist_matrix, sun_blocked


def _precompute_neighbors(planets, dist_matrix, sun_blocked):
    """For each planet, sorted list of nearest reachable (non-sun-blocked) targets."""
    neighbors = {}
    for p in planets:
        candidates = []
        for q in planets:
            if q.id == p.id:
                continue
            key = (p.id, q.id)
            if sun_blocked.get(key, False):
                continue
            candidates.append((dist_matrix.get(key, 9999.0), q.id))
        candidates.sort()
        neighbors[p.id] = [pid for _, pid in candidates]
    return neighbors


def _build_fleet_schedule(fleets, planets, step):
    """Convert existing fleets to (arrival_step, target_id, owner, ships) tuples."""
    schedule = []
    for f in fleets:
        best_planet = None
        best_eta = float("inf")
        for p in planets:
            eta = _fleet_hits_planet(f, p)
            if eta is not None and eta < best_eta:
                best_eta = eta
                best_planet = p
        if best_planet is not None:
            arrival = step + int(math.ceil(best_eta))
            schedule.append((arrival, best_planet.id, f.owner, int(f.ships)))
    return schedule


# ── simulation engine ──────────────────────────────────────────────────────

class _SimState:
    """Lightweight simulation state for forward projection."""
    __slots__ = ("planet_owner", "planet_ships", "planet_prod", "fleet_events",
                 "current_step", "planet_ids")

    def __init__(self, planet_owner, planet_ships, planet_prod, fleet_events,
                 current_step, planet_ids):
        self.planet_owner = planet_owner      # {pid: owner}
        self.planet_ships = planet_ships      # {pid: ships (float)}
        self.planet_prod = planet_prod        # {pid: production} (shared, not copied)
        self.fleet_events = fleet_events      # [(arrival_step, target_id, owner, ships)]
        self.current_step = current_step
        self.planet_ids = planet_ids          # list of all planet ids (shared)

    def copy(self):
        return _SimState(
            dict(self.planet_owner),
            dict(self.planet_ships),
            self.planet_prod,  # shared immutable
            list(self.fleet_events),
            self.current_step,
            self.planet_ids,  # shared immutable
        )


def _build_initial_state(planets, fleet_schedule, step):
    """Create initial SimState from parsed planets and fleet schedule."""
    planet_owner = {}
    planet_ships = {}
    planet_prod = {}
    planet_ids = []
    for p in planets:
        planet_owner[p.id] = p.owner
        planet_ships[p.id] = float(p.ships)
        planet_prod[p.id] = p.production
        planet_ids.append(p.id)
    return _SimState(planet_owner, planet_ships, planet_prod,
                     list(fleet_schedule), step, planet_ids)


def _sim_step(state):
    """Advance simulation by 1 step: production, then fleet arrivals + combat."""
    state.current_step += 1

    # Production for all owned planets
    for pid in state.planet_ids:
        if state.planet_owner[pid] >= 0:
            state.planet_ships[pid] += state.planet_prod[pid]

    # Collect arriving fleets this step
    arrivals = {}  # {target_id: {owner: total_ships}}
    remaining_events = []
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
            attackers[defender] = attackers.get(defender, 0) + garrison
        else:
            # Neutral planet: garrison fights as a separate "faction" (-1)
            attackers[-1] = attackers.get(-1, 0) + garrison

        # Find top two
        sorted_forces = sorted(attackers.items(), key=lambda x: -x[1])
        if len(sorted_forces) == 0:
            continue

        top_owner, top_ships = sorted_forces[0]
        second_ships = sorted_forces[1][1] if len(sorted_forces) > 1 else 0

        survivors = top_ships - second_ships
        if survivors <= 0:
            # Tie or no survivors — planet becomes neutral with 0 ships
            state.planet_owner[target_id] = -1
            state.planet_ships[target_id] = 0
        elif top_owner == defender:
            # Defender holds
            state.planet_ships[target_id] = survivors
        else:
            # Attacker captures
            state.planet_owner[target_id] = top_owner
            state.planet_ships[target_id] = survivors


# ── greedy rollout sub-policy ──────────────────────────────────────────────

def _rollout_substep(state, player, dist_matrix, neighbors, planets_by_id):
    """Greedy sub-policy with committed tracking.

    Same aggressive 70% sends as v1, but avoids wasteful double-targeting
    by tracking which planets already have fleets heading to them.
    """
    my_planets = [(pid, state.planet_ships[pid])
                  for pid in state.planet_ids
                  if state.planet_owner[pid] == player and state.planet_ships[pid] > 3]
    my_planets.sort(key=lambda x: -x[1])

    if not my_planets:
        return

    # Precompute committed ships per target (from fleet events)
    committed = {}
    for _, target_id, owner, ships in state.fleet_events:
        if owner == player:
            committed[target_id] = committed.get(target_id, 0) + ships

    targeted_this_step = set()
    launches = 0

    for src_id, src_ships in my_planets[:5]:
        if launches >= ROLLOUT_MAX_LAUNCHES:
            break

        best_score = -1.0
        best_target = None
        best_travel = 0

        for target_id in neighbors.get(src_id, [])[:ROLLOUT_NEIGHBORS]:
            if state.planet_owner[target_id] == player:
                continue
            if target_id in targeted_this_step:
                continue

            d = dist_matrix.get((src_id, target_id), 9999.0)
            speed = _fleet_speed(src_ships * 0.7)
            travel = d / max(speed, 0.1)

            target_ships = state.planet_ships[target_id]
            target_prod = state.planet_prod[target_id]
            target_owner = state.planet_owner[target_id]

            # Ships needed to capture, accounting for already committed
            already = committed.get(target_id, 0)
            prod_during = target_prod * travel if target_owner >= 0 else 0
            needed = target_ships + prod_during + 1 - already
            if needed < 1:
                continue  # Already committed enough

            if src_ships * 0.7 < needed:
                continue

            # ROI: production^2 / travel_time (same as v1)
            score = target_prod ** 2 / max(travel, 1.0)
            if score > best_score:
                best_score = score
                best_target = target_id
                best_travel = travel

        if best_target is not None:
            send = int(src_ships * 0.7)
            if send < 1:
                continue
            state.planet_ships[src_id] -= send
            arrival = state.current_step + max(1, int(math.ceil(best_travel)))
            state.fleet_events.append((arrival, best_target, player, send))
            committed[best_target] = committed.get(best_target, 0) + send
            targeted_this_step.add(best_target)
            launches += 1


def _enemy_decay(state, player):
    """Lightweight enemy activity model.

    Every 5 sim steps, enemy planets lose 50% of ships (models them
    sending fleets without tracking where). Cheaper and less noisy
    than running a full enemy rollout.
    """
    if state.current_step % 5 != 0:
        return
    for pid in state.planet_ids:
        owner = state.planet_owner[pid]
        if owner >= 0 and owner != player:
            state.planet_ships[pid] *= 0.5


# ── candidate generation & evaluation ──────────────────────────────────────

def _generate_candidates(src_id, state, player, dist_matrix, neighbors):
    """Generate WAIT + ATTACK candidates for a source planet.

    Returns list of (action_type, target_id, ships_to_send) tuples.
    'WAIT' means do nothing from this planet.
    """
    candidates = [("WAIT", None, 0)]

    src_ships = state.planet_ships[src_id]
    if src_ships < 2:
        return candidates

    targets_checked = 0
    for target_id in neighbors.get(src_id, []):
        if targets_checked >= MAX_TARGETS_PER_SOURCE:
            break
        if state.planet_owner[target_id] == player:
            continue

        targets_checked += 1

        d = dist_matrix.get((src_id, target_id), 9999.0)
        target_ships = state.planet_ships[target_id]
        target_prod = state.planet_prod[target_id]
        target_owner = state.planet_owner[target_id]

        # Compute "just enough" send amount for this target
        speed_est = _fleet_speed(src_ships * 0.5)
        travel_est = d / max(speed_est, 0.1)
        prod_during_est = target_prod * travel_est if target_owner >= 0 else 0
        just_enough = int((target_ships + prod_during_est + 1) * 1.3) + 1

        # Build send amounts: fixed fractions + just-enough
        send_amounts = set()
        for frac in SHIP_FRACTIONS:
            s = int(src_ships * frac)
            if s >= 1:
                send_amounts.add(s)
        if 1 <= just_enough <= src_ships:
            send_amounts.add(just_enough)

        for send in sorted(send_amounts, reverse=True):
            # Quick feasibility check
            speed = _fleet_speed(send)
            travel = d / max(speed, 0.1)
            prod_during = target_prod * travel if target_owner >= 0 else 0
            needed = target_ships + prod_during + 1

            if send < needed * 0.5:
                continue  # Skip clearly hopeless attacks

            candidates.append(("ATTACK", target_id, send))

    return candidates


def _evaluate_state(state, player):
    """Evaluate a simulation state for a player.

    Score = ships + fleets + production advantage bonus.
    Production advantage over enemies is weighted heavily because it compounds.
    """
    ships_on_planets = 0.0
    total_production = 0.0
    enemy_production = 0.0
    enemy_ships = 0.0
    n_my_planets = 0
    for pid in state.planet_ids:
        owner = state.planet_owner[pid]
        if owner == player:
            ships_on_planets += state.planet_ships[pid]
            total_production += state.planet_prod[pid]
            n_my_planets += 1
        elif owner >= 0:
            enemy_production += state.planet_prod[pid]
            enemy_ships += state.planet_ships[pid]

    ships_in_fleets = 0.0
    for event in state.fleet_events:
        _, _, owner, ships = event
        if owner == player:
            ships_in_fleets += ships

    remaining = max(0, MAX_STEPS - state.current_step)

    # Base: own ships (on planets + in fleets)
    score = ships_on_planets + ships_in_fleets

    # Production bonus: own production projected forward
    score += total_production * min(15.0, remaining / 8.0)

    # Production advantage bonus: dominating production compounds
    prod_advantage = total_production - enemy_production
    score += prod_advantage * min(8.0, remaining / 15.0)

    # Planet count bonus: having more planets = more options
    score += n_my_planets * 2.0

    return score


def _evaluate_candidate(state, src_id, ctype, target_id, send, player,
                        dist_matrix, neighbors, planets_by_id, depth):
    """Simulate a candidate action and return its score."""
    sim = state.copy()
    if ctype == "ATTACK":
        sim.planet_ships[src_id] -= send
        d = dist_matrix.get((src_id, target_id), 9999.0)
        speed = _fleet_speed(send)
        travel = max(1, int(math.ceil(d / max(speed, 0.1))))
        arrival = sim.current_step + travel
        sim.fleet_events.append((arrival, target_id, player, send))

    for _ in range(depth):
        _sim_step(sim)
        _rollout_substep(sim, player, dist_matrix, neighbors, planets_by_id)
        _enemy_decay(sim, player)

    return _evaluate_state(sim, player)


def _apply_decision(state, src_id, target_id, send, player, dist_matrix):
    """Apply an ATTACK decision to the shared state."""
    state.planet_ships[src_id] -= send
    d = dist_matrix.get((src_id, target_id), 9999.0)
    speed = _fleet_speed(send)
    travel = max(1, int(math.ceil(d / max(speed, 0.1))))
    arrival = state.current_step + travel
    state.fleet_events.append((arrival, target_id, player, send))


MIN_SECOND_PASS_SHIPS = 10


def _run_lookahead(state, player, dist_matrix, neighbors, planets_by_id, depth, time_limit):
    """Two-pass sequential greedy loop.

    Pass 1: Each of the top planets picks its best action (WAIT or ATTACK).
    Pass 2: Planets that sent <100% of their ships get a second chance to
    contribute with remaining ships — enables force splitting.

    Returns list of (src_id, target_id, ships) decisions.
    """
    start_time = time.time()
    decisions = []

    MAX_SIMULATED = 8  # Max attack candidates to simulate per planet

    def _best_action_for(src_id, eval_depth):
        """Evaluate candidates for src_id, return best (ctype, target_id, send).

        Pre-ranks attacks by a fast heuristic, then only simulates the top
        MAX_SIMULATED candidates + WAIT.
        """
        candidates = _generate_candidates(src_id, state, player, dist_matrix, neighbors)
        if len(candidates) <= 1:
            return candidates[0]

        # Pre-rank attacks by heuristic: production^2 / travel
        attacks = []
        for ctype, target_id, send in candidates:
            if ctype == "WAIT":
                continue
            d = dist_matrix.get((src_id, target_id), 9999.0)
            speed = _fleet_speed(send)
            travel = d / max(speed, 0.1)
            prod = state.planet_prod[target_id]
            heuristic = prod ** 2 / max(travel, 1.0)
            attacks.append((heuristic, ctype, target_id, send))

        attacks.sort(reverse=True)

        # Simulate WAIT + top N attacks
        to_simulate = [("WAIT", None, 0)]
        for _, ctype, target_id, send in attacks[:MAX_SIMULATED]:
            to_simulate.append((ctype, target_id, send))

        best_score = -float("inf")
        best_candidate = to_simulate[0]

        for ctype, target_id, send in to_simulate:
            score = _evaluate_candidate(
                state, src_id, ctype, target_id, send,
                player, dist_matrix, neighbors, planets_by_id, eval_depth)
            if score > best_score:
                best_score = score
                best_candidate = (ctype, target_id, send)

        return best_candidate

    # ── Pass 1: primary actions ──
    my_planets = [(pid, state.planet_ships[pid])
                  for pid in state.planet_ids
                  if state.planet_owner[pid] == player and state.planet_ships[pid] > 1]
    my_planets.sort(key=lambda x: -x[1])

    second_pass_candidates = []

    for src_id, ships_before in my_planets[:MAX_SOURCE_PLANETS]:
        if time.time() - start_time > time_limit:
            break

        ctype, target_id, send = _best_action_for(src_id, depth)

        if ctype == "ATTACK":
            _apply_decision(state, src_id, target_id, send, player, dist_matrix)
            decisions.append((src_id, target_id, send))
            # If planet still has ships, consider it for second pass
            remaining = state.planet_ships[src_id]
            if remaining >= MIN_SECOND_PASS_SHIPS:
                second_pass_candidates.append(src_id)

    # ── Pass 2: utilize remaining ships ──
    # Time budget: use remaining time, up to 30% of original limit
    pass2_limit = min(time_limit * 0.3, time_limit - (time.time() - start_time))

    if second_pass_candidates and pass2_limit > 0.05:
        pass2_start = time.time()
        # Use reduced depth for speed
        pass2_depth = max(5, depth // 2)

        for src_id in second_pass_candidates:
            if time.time() - pass2_start > pass2_limit:
                break
            if state.planet_ships[src_id] < MIN_SECOND_PASS_SHIPS:
                continue

            ctype, target_id, send = _best_action_for(src_id, pass2_depth)

            if ctype == "ATTACK":
                _apply_decision(state, src_id, target_id, send, player, dist_matrix)
                decisions.append((src_id, target_id, send))

    return decisions


# ── main agent function (MUST be last callable) ───────────────────────────

def agent(obs, config=None):
    """
    Lookahead agent for Orbit Wars.
    Uses forward simulation to evaluate moves by projecting game state N steps
    into the future with a greedy rollout sub-policy.
    """
    t_start = time.time()

    # ── parse observation ──────────────────────────────────────────────
    player = _get(obs, "player", 0)
    raw_planets = _get(obs, "planets", [])
    raw_fleets = _get(obs, "fleets", [])
    step = _get(obs, "step", 0)
    angular_velocity = _get(obs, "angular_velocity", 0.0)
    raw_initial = _get(obs, "initial_planets", [])
    overage_time = _get(obs, "remainingOverageTime", 60.0)

    planets = [_parse_planet(p) for p in raw_planets]
    fleets = [_parse_fleet(f) for f in raw_fleets]

    _detect_orbiting(planets, raw_initial, angular_velocity)

    planets_by_id = {p.id: p for p in planets}
    my_planets = [p for p in planets if p.owner == player]

    if not my_planets:
        return []

    # ── precomputation ────────────────────────────────────────────────
    dist_matrix, sun_blocked = _precompute_distances(planets)
    neighbors = _precompute_neighbors(planets, dist_matrix, sun_blocked)
    fleet_schedule = _build_fleet_schedule(fleets, planets, step)

    # ── adaptive depth based on remaining overage time ────────────────
    depth = LOOKAHEAD_DEPTH
    if overage_time < 20.0:
        depth = 10
    if overage_time < 10.0:
        depth = 5

    # Time budget: 800ms normally, less if overage is low
    time_limit = 0.8
    if overage_time < 15.0:
        time_limit = 0.3
    if overage_time < 5.0:
        time_limit = 0.1

    # ── build simulation state and run lookahead ──────────────────────
    sim_state = _build_initial_state(planets, fleet_schedule, step)
    decisions = _run_lookahead(sim_state, player, dist_matrix, neighbors,
                               planets_by_id, depth, time_limit)

    # ── convert decisions to actual moves with precise aiming ─────────
    moves = []
    for src_id, target_id, send in decisions:
        src = planets_by_id.get(src_id)
        target = planets_by_id.get(target_id)
        if src is None or target is None:
            continue

        result = _aim_at_moving(src, target, step, send)
        if result is not None:
            angle, _ = result
            moves.append([src_id, angle, send])

    return moves


if __name__ == "__main__":
    from kaggle_environments import make
    env = make("orbit_wars", debug=True)
    env.run([agent, "random"])
    final = env.steps[-1]
    for i, s in enumerate(final):
        print(f"Player {i}: reward={s.reward}, status={s.status}")
