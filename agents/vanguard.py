"""
Vanguard — improved strategic agent for Orbit Wars.

Improvements over strategic.py:
  - Coordinated multi-planet attacks (combine forces from multiple sources)
  - Time-weighted production scoring (early captures valued more)
  - Combat-aware threat calculation (top-vs-second rule, not sum)
  - Comet awareness (capture comets after spawn)
  - Endgame consolidation (step 350+: strip rear, consolidate core)
  - ROI-based early game (rank by production / cost)
  - No-attack discipline (skip targets with nearby enemy reinforcement)
  - Phase-adaptive redistribution (aggressive rear stripping late game)
"""

import math

# ── constants ──────────────────────────────────────────────────────────────
SUN_X, SUN_Y, SUN_RADIUS = 50.0, 50.0, 10.0
SUN_SAFE_RADIUS = SUN_RADIUS + 2.0
MAX_SHIP_SPEED = 6.0
BOARD_SIZE = 100.0
MIN_GARRISON = 3
BUFFER_FACTOR = 1.15


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


def _ships_needed(target_ships, target_owner, target_production, travel_turns):
    """Ships needed to capture, with buffer."""
    prod_during = target_production * travel_turns if target_owner >= 0 else 0
    raw = target_ships + prod_during + 1
    return int(math.ceil(raw * BUFFER_FACTOR))


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
        angle, is_detour = _safe_angle(src.x, src.y, target.x, target.y)
        tt = _travel_time(src.x, src.y, target.x, target.y, ships)
        return angle, tt

    tx, ty = target.x, target.y
    tt = _travel_time(src.x, src.y, tx, ty, ships)

    for _ in range(3):
        arrival = step + tt
        tx, ty = _planet_pos_at(target, arrival, step)
        tt = _travel_time(src.x, src.y, tx, ty, ships)

    angle, is_detour = _safe_angle(src.x, src.y, tx, ty)
    return angle, tt


# ── threat analysis (combat-aware) ────────────────────────────────────────

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

    # Check sun intersection
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
    """Returns {planet_id: [(owner, ships, eta), ...]} for incoming enemy fleets.

    Combat-aware: stores owner so we can simulate top-vs-second rule.
    """
    threats = {}
    enemy_fleets = [f for f in fleets if f.owner != player]

    for p in my_planets:
        for f in enemy_fleets:
            eta = _fleet_hits_planet(f, p)
            if eta is not None and eta < 50:
                if p.id not in threats:
                    threats[p.id] = []
                threats[p.id].append((f.owner, f.ships, eta))

    for pid in threats:
        threats[pid].sort(key=lambda x: x[2])

    return threats


def _garrison_needed(planet, threats):
    """Minimum ships to hold a planet based on incoming threats.

    Uses combat-aware calculation: top-vs-second rule for multiple attackers.
    """
    if planet.id not in threats:
        return MIN_GARRISON

    # Group threats arriving within 15 turns by owner
    imminent = [(owner, ships) for owner, ships, eta in threats[planet.id] if eta < 15]
    if not imminent:
        return MIN_GARRISON

    # Sum ships per attacking player
    by_owner = {}
    for owner, ships in imminent:
        by_owner[owner] = by_owner.get(owner, 0) + ships

    # Simulate combat: top attacker vs second (ties cancel)
    attackers = sorted(by_owner.values(), reverse=True)
    if len(attackers) == 1:
        # Single attacker: we need to beat them
        return max(MIN_GARRISON, attackers[0] + 1)
    else:
        # Multiple attackers fight each other first: top - second survive
        surviving_attack = attackers[0] - attackers[1]
        return max(MIN_GARRISON, surviving_attack + 1)


# ── strategic logic ────────────────────────────────────────────────────────

def _detect_players(planets, fleets, player):
    """Detect active players and their total production/ships."""
    stats = {}
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


def _enemy_centroid(enemy_planets):
    """Compute centroid of enemy planets for front/rear classification."""
    if not enemy_planets:
        return SUN_X, SUN_Y
    cx = sum(p.x for p in enemy_planets) / len(enemy_planets)
    cy = sum(p.y for p in enemy_planets) / len(enemy_planets)
    return cx, cy


def _has_nearby_enemy_reinforcement(target, enemy_planets, my_travel_time, player):
    """Check if enemy has planets nearby that can reinforce the target before we arrive."""
    for ep in enemy_planets:
        if ep.id == target.id:
            continue
        if ep.owner != target.owner:
            continue
        d = _dist(ep.x, ep.y, target.x, target.y)
        # Enemy reinforcement speed estimate (assume ~20 ships for speed calc)
        enemy_tt = d / _fleet_speed(20)
        # If enemy can reinforce before we arrive (with 2 turn margin)
        if enemy_tt < my_travel_time - 2 and ep.ships > 5:
            return True
    return False


# ── target scoring ─────────────────────────────────────────────────────────

def _target_score_early(src, target, step, ships_to_send):
    """ROI-based scoring for early game (steps 0-60).

    Score = production / (ships_needed * travel_time) — prioritize cheap nearby high-prod.
    """
    tt = _travel_time(src.x, src.y, target.x, target.y, ships_to_send)
    if tt < 0.1:
        tt = 0.1
    cost = ships_to_send * tt
    if cost < 1:
        cost = 1
    roi = target.production / cost
    # Bonus for neutrals (cheaper to capture, no production during travel)
    if target.owner == -1:
        roi *= 1.5
    return roi * 1000.0


def _target_score(src, target, step, player, n_players, leader_id, ships_to_send):
    """Score a target planet (higher = better).

    Uses time-weighted production value.
    """
    d = _dist(src.x, src.y, target.x, target.y)
    tt = _travel_time(src.x, src.y, target.x, target.y, ships_to_send)

    # Time-weighted production: early captures worth more
    remaining = max(1, 498 - step)
    prod_value = target.production * math.sqrt(remaining) * 0.5

    # Penalty for distance/travel time
    dist_penalty = d * 0.3 + tt * 2.0

    # Phase-based multiplier
    if target.owner == -1:
        if step < 60:
            type_mult = 2.0
        elif step < 200:
            type_mult = 1.0
        else:
            type_mult = 0.5
    else:
        if step < 60:
            type_mult = 0.5
        elif step < 200:
            type_mult = 1.0
        else:
            type_mult = 2.0

    # Bonus for attacking the leader in multiplayer
    leader_bonus = 0.0
    if n_players > 2 and target.owner == leader_id:
        leader_bonus = 15.0

    # Penalty for strong garrisons
    garrison_penalty = target.ships * 0.2

    score = (prod_value * type_mult + leader_bonus - dist_penalty - garrison_penalty)
    return score


# ── coordinated attack planner ─────────────────────────────────────────────

def _plan_coordinated_attacks(my_planets, all_planets, step, player, threats,
                              already_sent, n_players, leader_id, enemy_planets,
                              comet_ids):
    """Global target scoring with multi-planet coordination.

    For each target:
      - Find all planets that could contribute
      - If a single planet can handle it alone, prefer that (simpler)
      - Otherwise coordinate from multiple planets if arrival times are close
    """
    moves = []
    targets_claimed = {}  # target_id -> ships already committed

    # Build candidate targets
    targets = []
    for t in all_planets:
        if t.owner == player:
            continue
        targets.append(t)

    if not targets:
        return moves

    # Score targets globally (from centroid of our planets for ranking)
    my_cx = sum(p.x for p in my_planets) / len(my_planets)
    my_cy = sum(p.y for p in my_planets) / len(my_planets)

    target_scores = []
    for t in targets:
        # Use a "representative" score
        dummy_ships = 20
        if step < 60:
            score = _target_score_early(
                _Planet(-1, -1, my_cx, my_cy, 0, 0, 0), t, step, dummy_ships)
        else:
            score = _target_score(
                _Planet(-1, -1, my_cx, my_cy, 0, 0, 0), t, step, player,
                n_players, leader_id, dummy_ships)

        # Boost comets
        if t.id in comet_ids:
            score *= 1.5

        target_scores.append((score, t))

    target_scores.sort(key=lambda x: -x[0])

    # Process targets in priority order
    for _, target in target_scores:
        effective_ships = target.ships - targets_claimed.get(target.id, 0)
        if effective_ships < 0:
            effective_ships = 0

        # Find contributing planets sorted by travel time
        contributors = []
        for mp in my_planets:
            garrison = _garrison_needed(mp, threats)
            avail = mp.ships - already_sent.get(mp.id, 0) - garrison
            if avail < 3:
                continue
            tt = _travel_time(mp.x, mp.y, target.x, target.y, avail)
            contributors.append((tt, mp, avail))

        if not contributors:
            continue

        contributors.sort(key=lambda x: x[0])

        # Estimate ships needed
        best_tt = contributors[0][0]
        needed = _ships_needed(effective_ships, target.owner, target.production, best_tt)

        # No-attack discipline: skip if enemy can reinforce before us
        if target.owner >= 0 and _has_nearby_enemy_reinforcement(
                target, enemy_planets, best_tt, player):
            # Increase needed to account for reinforcement
            needed = int(needed * 1.5)

        # Try single-planet attack first (simpler, more reliable)
        single_sent = False
        for tt, mp, avail in contributors:
            needed_from_one = _ships_needed(effective_ships, target.owner,
                                            target.production, tt)
            if target.owner >= 0 and _has_nearby_enemy_reinforcement(
                    target, enemy_planets, tt, player):
                needed_from_one = int(needed_from_one * 1.5)

            if avail >= needed_from_one:
                ships_to_send = needed_from_one
                result = _aim_at_moving(mp, target, step, ships_to_send)
                if result is None:
                    continue
                angle, _ = result
                moves.append([mp.id, angle, ships_to_send])
                already_sent[mp.id] = already_sent.get(mp.id, 0) + ships_to_send
                targets_claimed[target.id] = targets_claimed.get(target.id, 0) + ships_to_send
                single_sent = True
                break

        if single_sent:
            continue

        # Multi-planet coordination: combine forces from planets with similar arrival times
        # Only coordinate from planets arriving within 3 turns of each other
        if len(contributors) < 2:
            continue

        # Take first contributor as reference
        ref_tt = contributors[0][0]
        coordinated = [(tt, mp, avail) for tt, mp, avail in contributors
                       if tt <= ref_tt + 3.0]

        total_available = sum(avail for _, _, avail in coordinated)
        # Recalculate needed based on latest arrival (worst case)
        max_tt = max(tt for tt, _, _ in coordinated)
        needed_coord = _ships_needed(effective_ships, target.owner,
                                     target.production, max_tt)

        if target.owner >= 0 and _has_nearby_enemy_reinforcement(
                target, enemy_planets, max_tt, player):
            needed_coord = int(needed_coord * 1.5)

        if total_available < needed_coord:
            continue

        # Send from each contributor proportionally
        remaining_need = needed_coord
        for tt, mp, avail in coordinated:
            if remaining_need <= 0:
                break
            send = min(avail, remaining_need)
            if send < 2:
                continue
            result = _aim_at_moving(mp, target, step, send)
            if result is None:
                continue
            angle, _ = result
            moves.append([mp.id, angle, send])
            already_sent[mp.id] = already_sent.get(mp.id, 0) + send
            targets_claimed[target.id] = targets_claimed.get(target.id, 0) + send
            remaining_need -= send

    return moves


# ── defense ────────────────────────────────────────────────────────────────

def _defense_moves(my_planets, threats, already_sent):
    """Reinforce planets under imminent threat from nearby friendly planets."""
    moves = []

    for p in my_planets:
        if p.id not in threats:
            continue

        imminent = [(ships, eta) for _, ships, eta in threats[p.id] if eta < 10]
        if not imminent:
            continue

        threat_ships = sum(s for s, _ in imminent)
        current = p.ships - already_sent.get(p.id, 0)
        deficit = threat_ships - current + 2

        if deficit <= 0:
            continue

        for donor in my_planets:
            if donor.id == p.id:
                continue
            donor_avail = donor.ships - already_sent.get(donor.id, 0) - _garrison_needed(donor, threats)
            if donor_avail < 3:
                continue

            d = _dist(donor.x, donor.y, p.x, p.y)
            tt = _travel_time(donor.x, donor.y, p.x, p.y, donor_avail)

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


# ── endgame consolidation ──────────────────────────────────────────────────

def _endgame_consolidation(my_planets, enemy_planets, threats, already_sent, step):
    """After step 350: strip isolated low-production planets, consolidate at core.

    Reduces expansion aggression and focuses on holding high-value territory.
    """
    moves = []

    if step < 350 or len(my_planets) < 3:
        return moves

    # Find high-production core planets (top 60% by production)
    sorted_by_prod = sorted(my_planets, key=lambda p: -p.production)
    core_count = max(2, int(len(sorted_by_prod) * 0.6))
    core_planets = set(p.id for p in sorted_by_prod[:core_count])

    # Enemy centroid for front/rear
    ecx, ecy = _enemy_centroid(enemy_planets)

    for p in my_planets:
        if p.id in core_planets:
            continue
        # Low-production non-core planet: strip aggressively
        garrison = _garrison_needed(p, threats)
        avail = p.ships - already_sent.get(p.id, 0) - garrison
        if avail < 5:
            continue

        # Send 80% of surplus to nearest core planet
        send = int(avail * 0.8)
        if send < 4:
            continue

        # Find nearest core planet that's toward the front
        best_core = None
        best_d = float("inf")
        for cp in my_planets:
            if cp.id not in core_planets:
                continue
            d = _dist(p.x, p.y, cp.x, cp.y)
            # Prefer core planets closer to enemy
            d_enemy = _dist(cp.x, cp.y, ecx, ecy)
            adjusted_d = d + d_enemy * 0.3
            if adjusted_d < best_d:
                best_d = adjusted_d
                best_core = cp

        if best_core is None:
            continue

        angle, _ = _safe_angle(p.x, p.y, best_core.x, best_core.y)
        moves.append([p.id, angle, send])
        already_sent[p.id] = already_sent.get(p.id, 0) + send

    return moves


# ── redistribution ─────────────────────────────────────────────────────────

def _redistribution_moves(my_planets, enemy_planets, threats, already_sent, step):
    """Move ships from rear to front. Phase-adaptive: more aggressive late game."""
    moves = []

    if len(my_planets) < 2 or not enemy_planets:
        return moves

    ecx, ecy = _enemy_centroid(enemy_planets)

    # Classify by distance to enemy centroid
    dists = [(p, _dist(p.x, p.y, ecx, ecy)) for p in my_planets]
    dists.sort(key=lambda x: x[1])

    median_idx = len(dists) // 2
    front = [p for p, d in dists[:median_idx + 1]]
    rear = [p for p, d in dists[median_idx + 1:]]

    if not rear or not front:
        return moves

    # Phase-adaptive surplus percentage
    if step >= 350:
        surplus_pct = 0.8  # aggressive stripping late game
        min_surplus = 4
    elif step >= 200:
        surplus_pct = 0.6
        min_surplus = 6
    else:
        surplus_pct = 0.5
        min_surplus = 8

    for rp in rear:
        garrison = _garrison_needed(rp, threats)
        avail = rp.ships - already_sent.get(rp.id, 0) - garrison
        if avail < min_surplus:
            continue

        send = int(avail * surplus_pct)
        if send < 4:
            continue

        # Send to nearest front planet
        best_front = min(front, key=lambda fp: _dist(rp.x, rp.y, fp.x, fp.y))

        angle, _ = _safe_angle(rp.x, rp.y, best_front.x, best_front.y)
        moves.append([rp.id, angle, send])
        already_sent[rp.id] = already_sent.get(rp.id, 0) + send

    return moves


# ── main agent function (MUST be last callable) ───────────────────────────

def agent(obs, config=None):
    """
    Vanguard — improved strategic Orbit Wars agent.
    Returns list of [from_planet_id, angle_radians, num_ships] moves.
    """
    # ── parse observation ──────────────────────────────────────────────
    player = _get(obs, "player", 0)
    raw_planets = _get(obs, "planets", [])
    raw_fleets = _get(obs, "fleets", [])
    step = _get(obs, "step", 0)
    angular_velocity = _get(obs, "angular_velocity", 0.0)
    raw_initial = _get(obs, "initial_planets", [])
    comet_planet_ids = _get(obs, "comet_planet_ids", [])

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
    player_stats = _detect_players(planets, fleets, player)
    n_players = len(player_stats) + 1
    leader_id = _identify_leader(player_stats)
    threats = _compute_threats(fleets, my_planets, player)

    # Comet IDs as a set for quick lookup
    comet_ids = set(comet_planet_ids) if comet_planet_ids else set()

    # ── generate moves in priority order ───────────────────────────────
    already_sent = {}

    # 1. Defense: reinforce threatened planets
    moves = _defense_moves(my_planets, threats, already_sent)

    # 2. Endgame consolidation (step 350+)
    if step >= 350:
        moves += _endgame_consolidation(my_planets, enemy_planets, threats,
                                        already_sent, step)

    # 3. Coordinated attacks (main offensive logic)
    moves += _plan_coordinated_attacks(
        my_planets, planets, step, player, threats, already_sent,
        n_players, leader_id, enemy_planets, comet_ids)

    # 4. Redistribution (if few attack moves)
    if len(moves) < 3:
        moves += _redistribution_moves(my_planets, enemy_planets, threats,
                                       already_sent, step)

    return moves


if __name__ == "__main__":
    from kaggle_environments import make
    env = make("orbit_wars", debug=True)
    env.run([agent, "random"])
    final = env.steps[-1]
    for i, s in enumerate(final):
        print(f"Player {i}: reward={s.reward}, status={s.status}")
