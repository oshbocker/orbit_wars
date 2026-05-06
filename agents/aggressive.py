"""
Aggressive production-rush agent for Orbit Wars.

Core insight: the player with the most ship production per turn almost always wins.
This agent uses target-first global planning to capture the highest-ROI planets ASAP.

Key features:
  - Target-first planning: rank all capturable planets by ROI, then assign sources
  - Multi-source coordinated attacks: multiple planets can attack the same target
  - ROI scoring: prod² / (cost x time) heavily favors high-production planets
  - Minimal garrison: only 2 ships base, scales with incoming threats
"""

import math

# ── constants ──────────────────────────────────────────────────────────────
SUN_X, SUN_Y, SUN_RADIUS = 50.0, 50.0, 10.0
SUN_SAFE_RADIUS = SUN_RADIUS + 2.0
MAX_SHIP_SPEED = 6.0
BOARD_SIZE = 100.0

MIN_GARRISON = 2        # bare minimum ships to keep
THREAT_GARRISON_MULT = 1.1  # hold 110% of incoming threat
OVERKILL = 1.2           # send 20% more than needed to capture
MIN_FLEET_SIZE = 3       # don't send tiny slow fleets


# ── helpers ────────────────────────────────────────────────────────────────

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
        # Check which planet this fleet will hit
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


# ── ROI scoring & target-first planning ────────────────────────────────────

def _ships_needed(target, travel_turns):
    """Ships needed to capture a target, with overkill buffer."""
    prod_during = target.production * travel_turns if target.owner >= 0 else 0
    raw = target.ships + prod_during + 1
    return max(MIN_FLEET_SIZE, int(math.ceil(raw * OVERKILL)))


# ── main agent function (MUST be last callable) ───────────────────────────

def agent(obs, config=None):
    """
    Aggressive production-rush agent for Orbit Wars.
    Target-first global planning: rank targets by ROI, coordinate multi-source attacks.
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

    # ── compute available ships per planet ─────────────────────────────
    available = {}
    for p in my_planets:
        garrison = _garrison_for(p, threats)
        available[p.id] = max(0, p.ships - garrison)

    # ── estimate friendly ships already in flight ──────────────────────
    in_flight = _estimate_in_flight(fleets, planets, player)

    # ── score all capturable targets by ROI ────────────────────────────
    targets = [p for p in planets if p.owner != player]

    scored = []
    for t in targets:
        # Find closest owned planet with ships to estimate travel time
        best_src = None
        best_dist = float("inf")
        for s in my_planets:
            if available[s.id] < MIN_FLEET_SIZE:
                continue
            d = _dist(s.x, s.y, t.x, t.y)
            if d < best_dist:
                best_dist = d
                best_src = s

        if best_src is None:
            continue

        # Estimate travel time with half of source's available ships
        est_ships = max(MIN_FLEET_SIZE, available[best_src.id] // 2)
        tt = _travel_time(best_src.x, best_src.y, t.x, t.y, est_ships)

        cost = _ships_needed(t, tt)
        already = in_flight.get(t.id, 0)
        remaining = max(1, cost - already)

        # ROI: production² / (remaining_cost × travel_time)
        # Higher production planets are exponentially more valuable
        roi = (t.production ** 2) / (remaining * max(tt, 1.0))

        scored.append((roi, t, remaining, already))

    # Sort by ROI descending
    scored.sort(key=lambda x: -x[0])

    # ── assign sources to targets (target-first) ──────────────────────
    moves = []

    for roi, target, needed, already in scored:
        if needed <= 0:
            # Already enough in-flight
            continue

        # Sort source planets by distance to this target
        sources = sorted(my_planets, key=lambda s: _dist(s.x, s.y, target.x, target.y))

        ships_still_needed = needed
        for src in sources:
            if available[src.id] < MIN_FLEET_SIZE:
                continue
            if ships_still_needed <= 0:
                break

            # Check sun blockage
            if _passes_through_sun(src.x, src.y, target.x, target.y):
                # Still try — _aim_at_moving uses _safe_angle for routing
                pass

            send = min(available[src.id], ships_still_needed)
            if send < MIN_FLEET_SIZE:
                continue

            # Aim with orbit prediction and sun avoidance
            result = _aim_at_moving(src, target, step, send)
            if result is None:
                continue
            angle, _ = result

            moves.append([src.id, angle, send])
            available[src.id] -= send
            ships_still_needed -= send

    return moves


if __name__ == "__main__":
    from kaggle_environments import make
    env = make("orbit_wars", debug=True)
    env.run([agent, "random"])
    final = env.steps[-1]
    for i, s in enumerate(final):
        print(f"Player {i}: reward={s.reward}, status={s.status}")
