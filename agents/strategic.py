"""
Strategic patient agent for Orbit Wars.

Core strategy: patience and overwhelming force.
  - Never send ships unless the source has enough to OVERWHELM the target
  - Wait and accumulate rather than sending dribbles
  - Accurate intercept solver for orbiting planets (accounts for spawn offset)
  - Single large fleets > multiple small ones (speed scales with fleet size)
"""

import math

# ── constants ──────────────────────────────────────────────────────────────
SUN_X, SUN_Y, SUN_RADIUS = 50.0, 50.0, 10.0
SUN_SAFE_RADIUS = SUN_RADIUS + 2.0
MAX_SHIP_SPEED = 6.0
BOARD_SIZE = 100.0

MIN_GARRISON = 0
THREAT_GARRISON_MULT = 1.2
OVERKILL = 1.0              # send 40% more than needed — overwhelming force
MIN_FLEET_SIZE = 0          # larger min to get meaningful speed
MAX_STEPS = 500

# ── helpers ────────────────────────────────────────────────────────────────

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


def _planet_pos_at(planet, future_step):
    """Predict planet position at a future step."""
    if not planet.is_orbiting:
        return planet.x, planet.y
    angle = planet.initial_angle + planet.angular_velocity * future_step
    x = SUN_X + planet.orbital_radius * math.cos(angle)
    y = SUN_Y + planet.orbital_radius * math.sin(angle)
    return x, y


def _aim_at_moving(src, target, step, ships):
    """Numerical intercept solver for orbiting targets.

    Solves the exact equation: find time t such that the fleet, launched now,
    arrives at the target planet's position at time (step + t) in exactly t turns.

    Uses bisection on f(t) = flight_time_to_target_pos(t) - t, which is guaranteed
    to have a root (fleet starts too far away, eventually catches up).

    Returns (angle, estimated_travel_time) or None if unreachable.
    """
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
        """f(t) = flight_time - t. Zero crossing = exact intercept."""
        # Target position at step + t
        a = target.initial_angle + omega * (step + t)
        tx = SUN_X + R * math.cos(a)
        ty = SUN_Y + R * math.sin(a)
        # Angle from source center to predicted position
        theta = math.atan2(ty - src.y, tx - src.x)
        # Fleet spawns at edge of source planet
        sx = src.x + src.radius * math.cos(theta)
        sy = src.y + src.radius * math.sin(theta)
        # Flight time at constant speed
        d = _dist(sx, sy, tx, ty)
        return d / speed - t

    # Phase 1: Coarse scan to find first bracket where f crosses zero
    # f(small_t) > 0 (fleet hasn't arrived), f(large_t) < 0 (fleet overshot)
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
        # No intercept found within max_t turns — aim at current position
        angle, _ = _safe_angle(src.x, src.y, target.x, target.y)
        spawn_x = src.x + src.radius * math.cos(angle)
        spawn_y = src.y + src.radius * math.sin(angle)
        tt = _travel_time(spawn_x, spawn_y, target.x, target.y, ships)
        return angle, tt

    # Phase 2: Bisection to find exact intercept time
    lo, hi = bracket
    for _ in range(25):  # converges to ~1e-7 precision
        mid = (lo + hi) / 2.0
        if _intercept_error(mid) > 0:
            lo = mid
        else:
            hi = mid

    t_intercept = (lo + hi) / 2.0

    # Compute the aim point at intercept time
    a = target.initial_angle + omega * (step + t_intercept)
    tx = SUN_X + R * math.cos(a)
    ty = SUN_Y + R * math.sin(a)

    # Check if path to intercept point passes through sun
    angle = math.atan2(ty - src.y, tx - src.x)
    spawn_x = src.x + src.radius * math.cos(angle)
    spawn_y = src.y + src.radius * math.sin(angle)

    if not _passes_through_sun(spawn_x, spawn_y, tx, ty):
        tt = _dist(spawn_x, spawn_y, tx, ty) / speed
        return angle, tt

    # Sun blocks this intercept — search for the next one (planet comes around)
    prev_t = bracket[1]
    prev_err = _intercept_error(prev_t)
    t = prev_t + dt
    while t <= max_t:
        err = _intercept_error(t)
        if prev_err >= 0 and err <= 0:
            # Found another bracket — bisect it
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

    # All intercepts blocked by sun — use sun-avoidance detour to first intercept
    a = target.initial_angle + omega * (step + (bracket[0] + bracket[1]) / 2.0)
    tx = SUN_X + R * math.cos(a)
    ty = SUN_Y + R * math.sin(a)
    angle, _ = _safe_angle(src.x, src.y, tx, ty)
    spawn_x = src.x + src.radius * math.cos(angle)
    spawn_y = src.y + src.radius * math.sin(angle)
    tt = _dist(spawn_x, spawn_y, tx, ty) / speed
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
    """Ships needed to capture a target, with overkill buffer.

    For neutral planets (owner == -1), no production during travel.
    """
    prod_during = target.production * travel_turns if target.owner >= 0 else 0
    raw = target.ships + prod_during + 1
    return max(MIN_FLEET_SIZE, int(math.ceil(raw * OVERKILL)))


# ── scoring ────────────────────────────────────────────────────────────────

def _score_target(src, target, step, ships_to_send):
    """Score a target by ROI: production^2 / (cost * time).

    Higher is better.
    """
    result = _aim_at_moving(src, target, step, ships_to_send)
    if result is None:
        return -1.0, 0.0, 0
    _, tt = result
    needed = _ships_needed(target, tt)
    if needed <= 0:
        return -1.0, tt, needed
    # Prefer high-production planets and closer ones (lower travel time)
    if ships_to_send >= needed:
        roi = target.production**2 / max(tt, 1.0)
    else:
        roi = 0
    return roi, tt, needed


# ── main agent function (MUST be last callable) ───────────────────────────

def agent(obs, config=None):
    """
    Strategic patient agent for Orbit Wars.
    Waits until it has enough ships to overwhelm a target, then sends one big fleet.
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

    # ── compute available ships per planet ─────────────────────────────
    available = {}
    for p in my_planets:
        garrison = _garrison_for(p, threats)
        available[p.id] = max(0, p.ships - garrison)

    # ── production advantage check ────────────────────────────────────
    my_production = sum(p.production for p in my_planets)
    enemy_planets = [p for p in planets if p.owner >= 0 and p.owner != player]
    enemy_production = sum(p.production for p in enemy_planets)

    # If we have a big production advantage, only target enemy planets
    if my_production >= enemy_production * 1.5 and enemy_planets:
        targets = enemy_planets
    else:
        targets = [p for p in planets if p.owner != player]

    moves = []
    used_sources = set()

    # ── for each of my planets, find the best target it can overwhelm ──
    # Sort my planets by available ships (descending) so the strongest attack first
    my_planets_sorted = sorted(my_planets, key=lambda p: -available[p.id])

    for src in my_planets_sorted:
        if src.id in used_sources:
            continue
        avail = available[src.id]
        if avail < MIN_FLEET_SIZE:
            continue

        # Score all targets from this source
        best_target = None
        best_roi = -1.0
        best_tt = 0.0
        best_needed = 0

        for target in targets:
            # Skip targets that already have enough ships in-flight
            already = in_flight.get(target.id, 0)
            if already > 0:
                # Check if in-flight is already enough
                # Rough estimate: if we've already sent enough, skip
                rough_tt = _travel_time(src.x, src.y, target.x, target.y, avail)
                rough_needed = _ships_needed(target, rough_tt)
                if already >= rough_needed:
                    continue

            roi, tt, needed = _score_target(src, target, step, avail)
            if roi < 0:
                continue

            # Subtract ships already in-flight toward this target
            already = in_flight.get(target.id, 0)
            effective_needed = max(MIN_FLEET_SIZE, needed - already)

            # KEY LOGIC: Only consider this target if we have enough ships NOW
            if avail < effective_needed:
                continue

            if roi > best_roi:
                best_roi = roi
                best_target = target
                best_tt = tt
                best_needed = effective_needed

        # If we found a target we can overwhelm, send the fleet
        if best_target is not None:
            # Send 90% of available ships — bigger fleets are faster and
            # more decisive. Never send less than what's needed.
            send = max(best_needed, int(avail * 0.9))
            result = _aim_at_moving(src, best_target, step, send)
            if result is not None:
                angle, _ = result
                moves.append([src.id, angle, send])
                used_sources.add(src.id)
                available[src.id] -= send
                # Update in-flight tracking for subsequent source decisions
                in_flight[best_target.id] = in_flight.get(best_target.id, 0) + send

    return moves


if __name__ == "__main__":
    from kaggle_environments import make
    env = make("orbit_wars", debug=True)
    env.run([agent, "random"])
    final = env.steps[-1]
    for i, s in enumerate(final):
        print(f"Player {i}: reward={s.reward}, status={s.status}")
