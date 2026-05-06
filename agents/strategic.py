"""
Strategic tree-search agent for Orbit Wars.

Core strategy: patience and calculation over speed.
  - Tree search (lookahead ~10 steps) to find optimal planet acquisition order
  - Ship consolidation before attacking — don't send ships until you can win
  - Three phases: early (local quadrant), mid (expand), late (assault enemy)
  - Massive late-game assaults exploit fleet speed scaling
"""

import math

# ── constants ──────────────────────────────────────────────────────────────
SUN_X, SUN_Y, SUN_RADIUS = 50.0, 50.0, 10.0
SUN_SAFE_RADIUS = SUN_RADIUS + 2.0
MAX_SHIP_SPEED = 6.0
BOARD_SIZE = 100.0

MIN_GARRISON = 2
THREAT_GARRISON_MULT = 1.1
OVERKILL = 1.25          # slightly more buffer since we send one big fleet
MIN_FLEET_SIZE = 5        # higher than aggressive — want speed
MAX_STEPS = 498

# Strategic constants
QUADRANT_SIZE = 25.0       # local search radius from home planet
LOOKAHEAD_STEPS = 10       # tree search depth in game steps
MAX_SEARCH_DEPTH = 4       # max sequential acquisitions to consider
LATE_GAME_STEP = 300       # switch to assault mode
MID_GAME_STEP = 80         # transition from early to mid
EXPAND_RADIUS = 60.0       # mid-game search radius


# ── helpers (shared with aggressive) ──────────────────────────────────────

def _dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def _fleet_speed(ships):
    if ships <= 1:
        return 1.0
    return 1.0 + (MAX_SHIP_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def _travel_time(src_x, src_y, dst_x, dst_y, ships):
    d = _dist(src_x, src_y, dst_x, dst_y)
    speed = _fleet_speed(ships)
    if speed <= 0:
        return 9999.0
    return d / speed


def _passes_through_sun(x1, y1, x2, y2):
    """Return True if the segment (x1,y1)->(x2,y2) passes within SUN_SAFE_RADIUS of the sun."""
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
    """Compute an angle from src to dst that avoids the sun.

    Returns (angle, is_detour).
    """
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
    """Mark planets that are orbiting based on initial_planets data."""
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


def _planet_pos_at(planet, future_step, current_step):
    """Predict planet position at a future step."""
    if not planet.is_orbiting:
        return planet.x, planet.y
    angle = planet.initial_angle + planet.angular_velocity * future_step
    x = SUN_X + planet.orbital_radius * math.cos(angle)
    y = SUN_Y + planet.orbital_radius * math.sin(angle)
    return x, y


def _aim_at_moving(src, target, step, ships):
    """Iterative intercept solver for orbiting targets.

    Returns (angle, estimated_travel_time) or None if unreachable.
    """
    if not target.is_orbiting:
        angle, _ = _safe_angle(src.x, src.y, target.x, target.y)
        tt = _travel_time(src.x, src.y, target.x, target.y, ships)
        return angle, tt

    tx, ty = target.x, target.y
    tt = _travel_time(src.x, src.y, tx, ty, ships)

    for _ in range(3):
        arrival = step + tt
        tx, ty = _planet_pos_at(target, arrival, step)
        tt = _travel_time(src.x, src.y, tx, ty, ships)

    angle, _ = _safe_angle(src.x, src.y, tx, ty)
    return angle, tt


# ── threat analysis ────────────────────────────────────────────────────────

def _fleet_hits_planet(fleet, planet):
    """Check if a fleet is heading toward a planet using line-circle intersection.

    Returns estimated turns to impact, or None if fleet won't hit.
    """
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

    # Check fleet doesn't cross sun first
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
    """Returns {planet_id: total_incoming_ships} for enemy fleets arriving within 15 turns."""
    threats = {}
    enemy_fleets = [f for f in fleets if f.owner != player]

    for p in my_planets:
        for f in enemy_fleets:
            eta = _fleet_hits_planet(f, p)
            if eta is not None and eta < 15:
                threats[p.id] = threats.get(p.id, 0) + f.ships

    return threats


def _garrison_for(planet, threats):
    """Minimum ships to keep on a planet."""
    threat_ships = threats.get(planet.id, 0)
    if threat_ships > 0:
        return max(MIN_GARRISON, int(math.ceil(threat_ships * THREAT_GARRISON_MULT)))
    return MIN_GARRISON


# ── in-flight tracking ─────────────────────────────────────────────────────

def _estimate_in_flight(fleets, planets, player):
    """Estimate how many friendly ships are already heading to each planet.

    Returns {planet_id: ships_in_flight}.
    """
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


def _ships_needed(target, travel_turns):
    """Ships needed to capture a target, with overkill buffer."""
    prod_during = target.production * travel_turns if target.owner >= 0 else 0
    raw = target.ships + prod_during + 1
    return max(MIN_FLEET_SIZE, int(math.ceil(raw * OVERKILL)))


# ── strategic functions ────────────────────────────────────────────────────

def _identify_home_planet(my_planets):
    """Return the planet with the highest production (tiebreak: most ships)."""
    if not my_planets:
        return None
    return max(my_planets, key=lambda p: (p.production, p.ships))


def _planets_in_region(planets, cx, cy, radius, player):
    """Return non-owned planets within radius of (cx, cy)."""
    result = []
    for p in planets:
        if p.owner == player:
            continue
        if _dist(p.x, p.y, cx, cy) <= radius:
            result.append(p)
    return result


def _detect_phase(step, my_planets, all_planets, player):
    """Determine the current game phase."""
    if step >= LATE_GAME_STEP:
        return "late"
    if step >= MID_GAME_STEP:
        return "mid"
    return "early"


class _SimState:
    """Lightweight simulation state for tree search."""
    __slots__ = ("owned_ids", "ships_available", "total_production",
                 "time_elapsed", "acquisitions", "score")

    def __init__(self, owned_ids, ships_available, total_production,
                 time_elapsed, acquisitions):
        self.owned_ids = set(owned_ids)
        self.ships_available = ships_available
        self.total_production = total_production
        self.time_elapsed = time_elapsed
        self.acquisitions = list(acquisitions)
        remaining = max(0, LOOKAHEAD_STEPS - time_elapsed)
        self.score = total_production * remaining + ships_available


def _tree_search_acquisitions(my_planets, candidates, step, planets_by_id):
    """DFS over planet acquisition sequences to maximize production.

    Returns [(target_id, ships_needed, estimated_launch_step), ...] or [].
    """
    if not my_planets or not candidates:
        return []

    # Initial state
    owned_ids = {p.id for p in my_planets}
    total_ships = sum(p.ships for p in my_planets)
    total_prod = sum(p.production for p in my_planets)

    initial = _SimState(owned_ids, total_ships, total_prod, 0, [])
    best_state = initial

    candidate_ids = [c.id for c in candidates]

    def _dfs(state, depth):
        nonlocal best_state
        if state.score > best_state.score:
            best_state = state
        if depth >= MAX_SEARCH_DEPTH or state.time_elapsed >= LOOKAHEAD_STEPS:
            return

        for cid in candidate_ids:
            if cid in state.owned_ids:
                continue
            target = planets_by_id.get(cid)
            if target is None:
                continue

            # Find closest owned planet to this target
            best_dist = float("inf")
            best_src = None
            for oid in state.owned_ids:
                src = planets_by_id.get(oid)
                if src is None:
                    continue
                d = _dist(src.x, src.y, target.x, target.y)
                if d < best_dist:
                    best_dist = d
                    best_src = src

            if best_src is None:
                continue

            # Estimate ships needed and travel time
            est_ships = max(MIN_FLEET_SIZE, int(state.ships_available * 0.5))
            tt = _travel_time(best_src.x, best_src.y, target.x, target.y, est_ships)
            needed = _ships_needed(target, tt)

            # How long must we wait to accumulate enough ships?
            if state.ships_available >= needed:
                wait = 0
            elif state.total_production > 0:
                deficit = needed - state.ships_available
                wait = math.ceil(deficit / state.total_production)
            else:
                continue  # can't afford it

            total_time = state.time_elapsed + wait + tt
            if total_time > LOOKAHEAD_STEPS:
                continue  # too slow

            # Simulate acquisition
            new_ships = state.ships_available + state.total_production * wait - needed
            # Ships accumulate during travel too (from remaining planets)
            new_ships += state.total_production * tt
            new_prod = state.total_production + target.production
            new_owned = state.owned_ids | {cid}
            new_acq = state.acquisitions + [(cid, needed, step + state.time_elapsed + wait)]

            new_state = _SimState(new_owned, new_ships, new_prod, total_time, new_acq)
            _dfs(new_state, depth + 1)

    _dfs(initial, 0)

    return best_state.acquisitions


def _find_staging_planet(my_planets, target):
    """Find the closest owned planet to the target."""
    if not my_planets:
        return None
    return min(my_planets, key=lambda p: _dist(p.x, p.y, target.x, target.y))


def _consolidation_moves(my_planets, staging, needed, threats, already_sent):
    """Generate moves to consolidate ships to a staging planet.

    Returns list of [from_planet_id, angle, ships] moves.
    """
    moves = []
    for p in my_planets:
        if p.id == staging.id:
            continue
        if p.id in already_sent:
            continue
        garrison = _garrison_for(p, threats)
        avail = p.ships - garrison
        if avail < MIN_FLEET_SIZE:
            continue
        angle, _ = _safe_angle(p.x, p.y, staging.x, staging.y)
        moves.append([p.id, angle, avail])
        already_sent.add(p.id)
    return moves


def _should_attack(staging, target, needed, in_flight_to_target, step):
    """Decide if we should launch an attack from staging to target."""
    garrison = MIN_GARRISON
    avail = staging.ships - garrison
    already = in_flight_to_target
    total = avail + already
    return total >= needed and avail >= MIN_FLEET_SIZE


def _late_game_moves(my_planets, enemy_planets, step, player, threats,
                     already_sent, fleets, planets, in_flight):
    """Generate late-game assault moves: target weakest enemy planets."""
    moves = []
    if not enemy_planets:
        return moves

    # Sort enemies by weakness (fewest ships, then lowest production)
    enemies_sorted = sorted(enemy_planets, key=lambda p: (p.ships, -p.production))

    for target in enemies_sorted:
        staging = _find_staging_planet(my_planets, target)
        if staging is None:
            break

        result = _aim_at_moving(staging, target, step, max(MIN_FLEET_SIZE, staging.ships))
        if result is None:
            continue
        _, tt = result

        needed = _ships_needed(target, tt)
        already = in_flight.get(target.id, 0)
        if already >= needed:
            continue

        remaining_needed = needed - already
        garrison = _garrison_for(staging, threats)
        avail = staging.ships - garrison

        if avail >= remaining_needed:
            # Launch the attack
            angle, _ = result
            send = remaining_needed
            moves.append([staging.id, angle, send])
            already_sent.add(staging.id)
        else:
            # Consolidate toward this staging planet
            consol = _consolidation_moves(my_planets, staging, remaining_needed,
                                          threats, already_sent)
            moves.extend(consol)
            # After consolidation, check if staging now has enough
            # (consolidation moves are in-flight, so we wait)
            break  # focus on one target at a time

    return moves


def _fallback_roi_attack(my_planets, all_planets, step, player, threats,
                         already_sent, fleets, in_flight):
    """Aggressive-style ROI attack as a safety net."""
    moves = []
    available = {}
    for p in my_planets:
        garrison = _garrison_for(p, threats)
        available[p.id] = max(0, p.ships - garrison)

    targets = [p for p in all_planets if p.owner != player]

    scored = []
    for t in targets:
        best_src = None
        best_d = float("inf")
        for s in my_planets:
            if available[s.id] < MIN_FLEET_SIZE or s.id in already_sent:
                continue
            d = _dist(s.x, s.y, t.x, t.y)
            if d < best_d:
                best_d = d
                best_src = s

        if best_src is None:
            continue

        est_ships = max(MIN_FLEET_SIZE, available[best_src.id] // 2)
        tt = _travel_time(best_src.x, best_src.y, t.x, t.y, est_ships)
        cost = _ships_needed(t, tt)
        already = in_flight.get(t.id, 0)
        remaining = max(1, cost - already)

        roi = (t.production ** 2) / (remaining * max(tt, 1.0))
        scored.append((roi, t, remaining, already, best_src))

    scored.sort(key=lambda x: -x[0])

    for roi, target, needed, already, best_src in scored:
        if needed <= 0:
            continue

        sources = sorted(my_planets,
                         key=lambda s: _dist(s.x, s.y, target.x, target.y))

        ships_still_needed = needed
        for src in sources:
            if available.get(src.id, 0) < MIN_FLEET_SIZE or src.id in already_sent:
                continue
            if ships_still_needed <= 0:
                break

            send = min(available[src.id], ships_still_needed)
            if send < MIN_FLEET_SIZE:
                continue

            result = _aim_at_moving(src, target, step, send)
            if result is None:
                continue
            angle, _ = result

            moves.append([src.id, angle, send])
            available[src.id] -= send
            already_sent.add(src.id)
            ships_still_needed -= send

    return moves


# ── main agent function (MUST be last callable) ───────────────────────────

def agent(obs, config=None):
    """
    Strategic tree-search agent for Orbit Wars.
    Uses lookahead planning, ship consolidation, and phased play.
    Returns list of [from_planet_id, angle_radians, num_ships] moves.
    """
    # ── parse observation ──────────────────────────────────────────────
    player = _get(obs, "player", 0)
    raw_planets = _get(obs, "planets", [])
    raw_fleets = _get(obs, "fleets", [])
    step = _get(obs, "step", 0)
    angular_velocity = _get(obs, "angular_velocity", 0.0)
    raw_initial = _get(obs, "initial_planets", [])

    planets = [_parse_planet(p) for p in raw_planets]
    fleets = [_parse_fleet(f) for f in raw_fleets]

    _detect_orbiting(planets, raw_initial, angular_velocity)

    planets_by_id = {p.id: p for p in planets}
    my_planets = [p for p in planets if p.owner == player]

    if not my_planets:
        return []

    # ── threat analysis ────────────────────────────────────────────────
    threats = _compute_threats(fleets, my_planets, player)

    # ── in-flight tracking ─────────────────────────────────────────────
    in_flight = _estimate_in_flight(fleets, planets, player)

    # ── phase detection ────────────────────────────────────────────────
    phase = _detect_phase(step, my_planets, planets, player)

    already_sent = set()  # track which planets we've issued orders from
    moves = []

    # ── early & mid game: tree search + consolidate + attack ───────────
    if phase in ("early", "mid"):
        home = _identify_home_planet(my_planets)
        if home is None:
            return []

        # Determine search center and radius
        if phase == "early":
            cx, cy = home.x, home.y
            search_radius = QUADRANT_SIZE
        else:
            # Mid game: use centroid of owned planets
            cx = sum(p.x for p in my_planets) / len(my_planets)
            cy = sum(p.y for p in my_planets) / len(my_planets)
            search_radius = EXPAND_RADIUS

        # Find candidate planets for acquisition
        candidates = _planets_in_region(planets, cx, cy, search_radius, player)

        # Run tree search
        plan = _tree_search_acquisitions(my_planets, candidates, step, planets_by_id)

        if plan:
            # Execute the first acquisition in the plan
            target_id, needed, launch_step = plan[0]
            target = planets_by_id.get(target_id)
            if target is not None:
                staging = _find_staging_planet(my_planets, target)
                if staging is not None:
                    already_to_target = in_flight.get(target_id, 0)
                    remaining_needed = max(1, needed - already_to_target)

                    if _should_attack(staging, target, remaining_needed,
                                      already_to_target, step):
                        # Launch attack
                        garrison = _garrison_for(staging, threats)
                        avail = staging.ships - garrison
                        send = min(avail, remaining_needed)
                        if send >= MIN_FLEET_SIZE:
                            result = _aim_at_moving(staging, target, step, send)
                            if result is not None:
                                angle, _ = result
                                moves.append([staging.id, angle, send])
                                already_sent.add(staging.id)
                    else:
                        # Consolidate ships toward staging planet
                        consol = _consolidation_moves(
                            my_planets, staging, remaining_needed,
                            threats, already_sent)
                        moves.extend(consol)

        # If tree search produced no moves, try fallback
        if not moves:
            moves = _fallback_roi_attack(
                my_planets, planets, step, player, threats,
                already_sent, fleets, in_flight)

    # ── late game: assault enemy planets ───────────────────────────────
    elif phase == "late":
        enemy_planets = [p for p in planets if p.owner >= 0 and p.owner != player]
        moves = _late_game_moves(
            my_planets, enemy_planets, step, player, threats,
            already_sent, fleets, planets, in_flight)

        # If late game produced no moves, fallback
        if not moves:
            moves = _fallback_roi_attack(
                my_planets, planets, step, player, threats,
                already_sent, fleets, in_flight)

    return moves


if __name__ == "__main__":
    from kaggle_environments import make
    env = make("orbit_wars", debug=True)
    env.run([agent, "random"])
    final = env.steps[-1]
    for i, s in enumerate(final):
        print(f"Player {i}: reward={s.reward}, status={s.status}")
