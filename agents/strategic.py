"""
Strategic rules-based agent for Orbit Wars.

Improvements over baseline:
  - Phase-based strategy (early expansion → balanced → aggressive)
  - Orbit prediction: aim at where moving planets will be when fleet arrives
  - Threat analysis: analytical line-circle intersection to detect incoming fleets
  - Dynamic garrison: hold ships proportional to incoming threats
  - Sun avoidance routing: waypoint-based routing around the sun
  - Multi-launch: send multiple fleets from one planet if surplus is large
  - Redistribution: move ships from rear to front-line planets when idle
  - 4-player awareness: focus attacks on the leader
"""

import math

# ── constants ──────────────────────────────────────────────────────────────
SUN_X, SUN_Y, SUN_RADIUS = 50.0, 50.0, 10.0
SUN_SAFE_RADIUS = SUN_RADIUS + 2.0
MAX_SHIP_SPEED = 6.0
BOARD_SIZE = 100.0
MIN_GARRISON = 3  # absolute minimum even with no threats
BUFFER_FACTOR = 1.15  # 15% over-send buffer


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
    """Return True if the segment (x1,y1)→(x2,y2) passes within SUN_SAFE_RADIUS of the sun."""
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


def _ships_needed(target_ships, target_owner, target_production, travel_turns):
    """Ships needed to capture, with buffer."""
    prod_during = target_production * travel_turns if target_owner >= 0 else 0
    raw = target_ships + prod_during + 1
    return int(math.ceil(raw * BUFFER_FACTOR))


def _safe_angle(src_x, src_y, dst_x, dst_y):
    """Compute an angle from src to dst that avoids the sun.

    If the direct path is clear, return the direct angle.
    Otherwise, route via a waypoint tangent to the sun safety radius.
    Returns (angle, is_detour).
    """
    direct = math.atan2(dst_y - src_y, dst_x - src_x)
    if not _passes_through_sun(src_x, src_y, dst_x, dst_y):
        return direct, False

    # Pick a waypoint on the perpendicular side of the sun
    # Try two tangent points and pick the one closer to the destination
    cx, cy = SUN_X, SUN_Y
    # Angle from sun center to source
    a_src = math.atan2(src_y - cy, src_x - cx)
    # Two candidate waypoints: ±90° from a_src, at SUN_SAFE_RADIUS + 3
    r = SUN_SAFE_RADIUS + 3.0
    best_wp = None
    best_total = float("inf")
    for offset in (math.pi / 2, -math.pi / 2):
        wp_a = a_src + offset
        wx = cx + r * math.cos(wp_a)
        wy = cy + r * math.sin(wp_a)
        # Clamp to board
        wx = max(1.0, min(99.0, wx))
        wy = max(1.0, min(99.0, wy))
        total = _dist(src_x, src_y, wx, wy) + _dist(wx, wy, dst_x, dst_y)
        if total < best_total and not _passes_through_sun(src_x, src_y, wx, wy):
            best_total = total
            best_wp = (wx, wy)

    if best_wp is None:
        # Fallback: try more angles
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
        return direct, False  # give up, send direct

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


def _get(obs, key, default=None):
    if hasattr(obs, key):
        return getattr(obs, key)
    if isinstance(obs, dict):
        return obs.get(key, default)
    return default


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
        # Get initial position
        if hasattr(ip, "x"):
            ix, iy = ip.x, ip.y
        elif isinstance(ip, dict):
            ix, iy = ip["x"], ip["y"]
        else:
            ix, iy = ip[2], ip[3]

        orb_r = _dist(ix, iy, SUN_X, SUN_Y)
        # Orbiting if orbital_radius + planet_radius < 50 (inside the board circle)
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
        angle, is_detour = _safe_angle(src.x, src.y, target.x, target.y)
        tt = _travel_time(src.x, src.y, target.x, target.y, ships)
        return angle, tt

    # Iterative: guess arrival time, predict position, refine
    tx, ty = target.x, target.y
    tt = _travel_time(src.x, src.y, tx, ty, ships)

    for _ in range(3):
        arrival = step + tt
        tx, ty = _planet_pos_at(target, arrival, step)
        tt = _travel_time(src.x, src.y, tx, ty, ships)

    angle, is_detour = _safe_angle(src.x, src.y, tx, ty)
    return angle, tt


# ── threat analysis ────────────────────────────────────────────────────────

def _fleet_hits_planet(fleet, planet):
    """Check if a fleet is heading toward a planet using line-circle intersection.

    Returns estimated turns to impact, or None if fleet won't hit.
    """
    speed = _fleet_speed(fleet.ships)
    # Fleet direction vector
    dx = math.cos(fleet.angle) * speed
    dy = math.sin(fleet.angle) * speed

    # Vector from fleet to planet center
    fx = fleet.x - planet.x
    fy = fleet.y - planet.y

    hit_r = planet.radius + 0.5  # small buffer for collision detection

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

    # We want the earliest future intersection
    t = t1 if t1 > 0 else t2
    if t <= 0:
        return None

    # t is in "turns" since a = speed^2 and we moved by speed per turn
    # Actually t is parametric along (dx, dy) per turn, so t = turns
    # But check the fleet doesn't go out of bounds first
    hit_x = fleet.x + dx * t
    hit_y = fleet.y + dy * t
    if hit_x < 0 or hit_x > BOARD_SIZE or hit_y < 0 or hit_y > BOARD_SIZE:
        # Fleet goes off board before reaching planet
        d_edge = min(
            (BOARD_SIZE - fleet.x) / max(dx, 1e-10) if dx > 0 else float("inf"),
            -fleet.x / min(dx, -1e-10) if dx < 0 else float("inf"),
            (BOARD_SIZE - fleet.y) / max(dy, 1e-10) if dy > 0 else float("inf"),
            -fleet.y / min(dy, -1e-10) if dy < 0 else float("inf"),
        )
        if t > d_edge:
            return None

    # Also check fleet doesn't cross sun first
    sun_fx = fleet.x - SUN_X
    sun_fy = fleet.y - SUN_Y
    sun_a = a  # same direction
    sun_b = 2 * (sun_fx * dx + sun_fy * dy)
    sun_c = sun_fx * sun_fx + sun_fy * sun_fy - SUN_RADIUS * SUN_RADIUS
    sun_disc = sun_b * sun_b - 4 * sun_a * sun_c
    if sun_disc >= 0:
        sun_sq = math.sqrt(sun_disc)
        sun_t = (-sun_b - sun_sq) / (2 * sun_a)
        if 0 < sun_t < t:
            return None  # Fleet hits sun before reaching planet

    return t


def _compute_threats(fleets, my_planets, player):
    """Returns {planet_id: [(ships, eta), ...]} for incoming enemy fleets."""
    threats = {}
    enemy_fleets = [f for f in fleets if f.owner != player]

    for p in my_planets:
        for f in enemy_fleets:
            eta = _fleet_hits_planet(f, p)
            if eta is not None and eta < 50:  # only care about threats within 50 turns
                if p.id not in threats:
                    threats[p.id] = []
                threats[p.id].append((f.ships, eta))

    # Sort by ETA
    for pid in threats:
        threats[pid].sort(key=lambda x: x[1])

    return threats


def _garrison_needed(planet, threats):
    """Minimum ships to hold a planet based on incoming threats."""
    if planet.id not in threats:
        return MIN_GARRISON

    # Sum ships of threats arriving within next 15 turns
    imminent = sum(s for s, eta in threats[planet.id] if eta < 15)
    return max(MIN_GARRISON, imminent + 1)


# ── strategic logic ────────────────────────────────────────────────────────

def _compute_phase(step):
    """Return phase multipliers based on game progression.

    Returns (neutral_bonus, enemy_bonus, expansion_aggression).
    """
    if step < 60:
        # Phase 1: Rapid neutral expansion
        return 2.0, 0.5, 1.5
    elif step < 200:
        # Phase 2: Balanced
        return 1.0, 1.0, 1.0
    else:
        # Phase 3: Aggressive enemy targeting
        return 0.5, 2.0, 0.8


def _detect_players(planets, fleets, player):
    """Detect active players and their total production/ships."""
    stats = {}  # player_id -> (total_production, total_ships)
    for p in planets:
        if p.owner >= 0 and p.owner != player:
            if p.owner not in stats:
                stats[p.owner] = [0, 0]
            stats[p.owner][0] += p.production
            stats[p.owner][1] += p.ships

    for f in fleets:
        if f.owner >= 0 and f.owner != player:
            if f.owner not in stats:
                stats[f.owner] = [0, 0]
            stats[f.owner][1] += f.ships

    return stats


def _identify_leader(player_stats):
    """Return the player_id of the leader (most production), or None."""
    if not player_stats:
        return None
    return max(player_stats, key=lambda pid: player_stats[pid][0])


def _compute_frontline(my_planets, enemy_planets):
    """Classify planets as frontline or rear.

    Returns {planet_id: min_distance_to_enemy}.
    """
    dist_to_enemy = {}
    for mp in my_planets:
        if not enemy_planets:
            dist_to_enemy[mp.id] = 999.0
        else:
            dist_to_enemy[mp.id] = min(_dist(mp.x, mp.y, ep.x, ep.y) for ep in enemy_planets)
    return dist_to_enemy


def _target_score(src, target, step, player, n_players, leader_id, phase_mults, ships_to_send):
    """Score a target planet (higher = better).

    Considers: production value, distance, ships needed, phase, leader bonus.
    """
    neutral_bonus, enemy_bonus, _ = phase_mults

    d = _dist(src.x, src.y, target.x, target.y)
    tt = _travel_time(src.x, src.y, target.x, target.y, ships_to_send)

    # Base: production value
    prod_value = target.production * 10.0

    # Penalty for distance/travel time
    dist_penalty = d * 0.3 + tt * 2.0

    # Phase-based multiplier
    if target.owner == -1:
        type_mult = neutral_bonus
    else:
        type_mult = enemy_bonus

    # Bonus for attacking the leader in multiplayer
    leader_bonus = 0.0
    if n_players > 2 and target.owner == leader_id:
        leader_bonus = 15.0

    # Bonus for orbiting planets near other orbiting planets (chain capture)
    orbit_bonus = 0.0
    if target.is_orbiting:
        orbit_bonus = 3.0  # slight bonus for orbiting targets (often higher production)

    # Penalty for strong garrisons
    garrison_penalty = target.ships * 0.2

    score = (prod_value * type_mult + leader_bonus + orbit_bonus
             - dist_penalty - garrison_penalty)

    return score


# ── decision functions ─────────────────────────────────────────────────────

def _defense_moves(my_planets, threats, already_sent, planets_by_id):
    """Reinforce planets under imminent threat from nearby friendly planets."""
    moves = []

    for p in my_planets:
        if p.id not in threats:
            continue

        # Total incoming threat within 10 turns
        imminent = [(s, eta) for s, eta in threats[p.id] if eta < 10]
        if not imminent:
            continue

        threat_ships = sum(s for s, _ in imminent)
        current = p.ships - already_sent.get(p.id, 0)
        deficit = threat_ships - current + 2  # need a small buffer to win combat

        if deficit <= 0:
            continue  # planet can hold on its own

        # Find nearby friendly planets that can send reinforcements
        for donor in my_planets:
            if donor.id == p.id:
                continue
            donor_avail = donor.ships - already_sent.get(donor.id, 0) - _garrison_needed(donor, threats)
            if donor_avail < 3:
                continue

            d = _dist(donor.x, donor.y, p.x, p.y)
            tt = _travel_time(donor.x, donor.y, p.x, p.y, donor_avail)

            # Only reinforce if help arrives in time
            min_eta = min(eta for _, eta in imminent)
            if tt > min_eta + 2:
                continue

            send = min(donor_avail, deficit)
            if send < 2:
                continue

            angle, _ = _safe_angle(donor.x, donor.y, p.x, p.y)
            moves.append([donor.id, angle, send])
            already_sent[donor.id] = already_sent.get(donor.id, 0) + send
            deficit -= send
            if deficit <= 0:
                break

    return moves


def _expansion_moves(my_planets, all_planets, step, player, threats, already_sent,
                     n_players, leader_id, phase_mults):
    """Main attack planner: score targets, send fleets."""
    moves = []
    targets_claimed = {}  # target_id -> total ships already committed

    # Sort own planets by production descending
    sorted_mine = sorted(my_planets, key=lambda p: -p.production)

    for mine in sorted_mine:
        garrison = _garrison_needed(mine, threats)
        available = mine.ships - already_sent.get(mine.id, 0) - garrison
        if available < 3:
            continue

        # Score all non-owned targets
        candidates = []
        for t in all_planets:
            if t.owner == player:
                continue
            # Adjust target ships for already-committed attacks
            effective_ships = t.ships - targets_claimed.get(t.id, 0)
            if effective_ships < 0:
                effective_ships = 0

            # Estimate ships needed
            tt = _travel_time(mine.x, mine.y, t.x, t.y, available)
            needed = _ships_needed(effective_ships, t.owner, t.production, tt)

            if needed > available:
                continue

            score = _target_score(mine, t, step, player, n_players, leader_id,
                                  phase_mults, needed)
            candidates.append((score, t, needed, tt))

        if not candidates:
            continue

        # Sort by score descending
        candidates.sort(key=lambda x: -x[0])

        # Try to send to the best target(s)
        for score, target, needed, tt in candidates:
            current_avail = mine.ships - already_sent.get(mine.id, 0) - garrison
            if current_avail < 3:
                break

            ships_to_send = min(needed, current_avail)
            if ships_to_send < 2:
                continue

            # Get angle (with orbit prediction and sun avoidance)
            result = _aim_at_moving(mine, target, step, ships_to_send)
            if result is None:
                continue
            angle, _ = result

            moves.append([mine.id, angle, ships_to_send])
            already_sent[mine.id] = already_sent.get(mine.id, 0) + ships_to_send
            targets_claimed[target.id] = targets_claimed.get(target.id, 0) + ships_to_send

            # Allow multiple launches from one planet, but check available again
            current_avail = mine.ships - already_sent.get(mine.id, 0) - garrison
            if current_avail < 5:
                break

    return moves


def _redistribution_moves(my_planets, enemy_planets, threats, already_sent):
    """Move ships from rear planets to front-line planets when no attack targets."""
    moves = []

    if len(my_planets) < 2 or not enemy_planets:
        return moves

    frontline_dist = _compute_frontline(my_planets, enemy_planets)

    # Identify rear planets (far from enemies) and front planets (close to enemies)
    median_dist = sorted(frontline_dist.values())[len(frontline_dist) // 2] if frontline_dist else 50.0

    rear = [p for p in my_planets if frontline_dist.get(p.id, 999) > median_dist]
    front = [p for p in my_planets if frontline_dist.get(p.id, 999) <= median_dist]

    if not rear or not front:
        return moves

    # Find best front-line planet to reinforce (lowest ships per production)
    front_sorted = sorted(front, key=lambda p: p.ships / max(p.production, 1))

    for rp in rear:
        garrison = _garrison_needed(rp, threats)
        available = rp.ships - already_sent.get(rp.id, 0) - garrison
        if available < 8:  # only redistribute if significant surplus
            continue

        # Send to nearest front-line planet
        best_front = min(front_sorted[:3], key=lambda fp: _dist(rp.x, rp.y, fp.x, fp.y))
        send = available // 2  # send half the surplus

        if send < 5:
            continue

        angle, _ = _safe_angle(rp.x, rp.y, best_front.x, best_front.y)
        moves.append([rp.id, angle, send])
        already_sent[rp.id] = already_sent.get(rp.id, 0) + send

    return moves


# ── main agent function (MUST be last callable) ───────────────────────────

def agent(obs, config=None):
    """
    Strategic Orbit Wars agent.
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

    # Detect orbiting planets
    _detect_orbiting(planets, raw_initial, angular_velocity)

    planets_by_id = {p.id: p for p in planets}
    my_planets = [p for p in planets if p.owner == player]
    enemy_planets = [p for p in planets if p.owner >= 0 and p.owner != player]

    if not my_planets:
        return []

    # ── strategic analysis ─────────────────────────────────────────────
    phase_mults = _compute_phase(step)
    player_stats = _detect_players(planets, fleets, player)
    n_players = len(player_stats) + 1  # +1 for us
    leader_id = _identify_leader(player_stats)
    threats = _compute_threats(fleets, my_planets, player)

    # ── generate moves in priority order ───────────────────────────────
    already_sent = {}  # planet_id -> ships committed this turn

    # 1. Defense: reinforce threatened planets
    moves = _defense_moves(my_planets, threats, already_sent, planets_by_id)

    # 2. Expansion: attack best targets
    moves += _expansion_moves(my_planets, planets, step, player, threats,
                              already_sent, n_players, leader_id, phase_mults)

    # 3. Redistribution: move rear ships to front (only if few expansion moves)
    if len(moves) < 2:
        moves += _redistribution_moves(my_planets, enemy_planets, threats, already_sent)

    return moves


if __name__ == "__main__":
    from kaggle_environments import make
    env = make("orbit_wars", debug=True)
    env.run([agent, "random"])
    final = env.steps[-1]
    for i, s in enumerate(final):
        print(f"Player {i}: reward={s.reward}, status={s.status}")
