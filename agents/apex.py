"""
Apex agent for Orbit Wars.

Merges the best ideas from the hybrid agent and the 1103 peaking bot:
- All hybrid features: beam search opening, waypoint sun avoidance, timeline
  simulation, mission-based planning, multi-source coordination
- Exposed-enemy bonus: detects weakened enemy planets (from peaking bot)
- Crash exploit detection: strikes after enemy fleets destroy each other
- Recapture missions: plans to take back planets after they fall
- Stacked enemy threat reserves: better defense against coordinated attacks
- Tuned constants for more aggressive play
"""

import math
import time
from collections import defaultdict

# ── constants ─────────────────────────────────────────────────────────────
BOARD = 100.0
SUN_X, SUN_Y, SUN_R = 50.0, 50.0, 10.0
SUN_SAFE = 1.5
MAX_SPEED = 6.0
MAX_STEPS = 500
ORBIT_LIMIT = 50.0
LAUNCH_OFFSET = 0.1
HORIZON = 110

# Time management
SOFT_DEADLINE_FRAC = 0.82
HEAVY_PHASE_MIN_TIME = 0.16
OPTIONAL_PHASE_MIN_TIME = 0.08

# Mission value multipliers
NEUTRAL_VALUE_MULT = 1.4
HOSTILE_VALUE_MULT = 2.2
OPENING_HOSTILE_MULT = 1.6
STATIC_BONUS = 1.22
SNIPE_BONUS = 1.15
SWARM_BONUS = 1.08
REINFORCE_BONUS = 1.35
CRASH_EXPLOIT_BONUS = 1.25
COMET_PENALTY = 0.85
LEADER_BONUS = 1.2
EXPOSED_ENEMY_BONUS = 0.25

# Ship margins (aggressively tight — key efficiency advantage)
NEUTRAL_MARGIN_BASE = 1
NEUTRAL_MARGIN_PROD = 1
NEUTRAL_MARGIN_CAP = 5
HOSTILE_MARGIN_BASE = 1
HOSTILE_MARGIN_PROD = 1
HOSTILE_MARGIN_CAP = 7
STATIC_MARGIN_BONUS = 2  # hybrid uses 4
LONG_TRIP_DIVISOR = 5  # hybrid uses 3

# Defense
DEFENSE_LOOKAHEAD = 28
DEFENSE_MARGIN_BASE = 1
DEFENSE_MARGIN_PROD = 1
PROACTIVE_DEFENSE_HORIZON = 12
PROACTIVE_DEFENSE_RATIO = 0.12
STACKED_THREAT_WINDOW = 3
STACKED_THREAT_RATIO = 0.18

# Reinforce
REINFORCE_MIN_PROD = 2
REINFORCE_MAX_TRAVEL = 22
REINFORCE_MAX_SRC_FRAC = 0.75
REINFORCE_MIN_FUTURE = 25
REINFORCE_HOLD_LOOKAHEAD = 20
REINFORCE_SAFETY = 2

# Swarm (more permissive than hybrid for better coordination)
MIN_PARTIAL_SHIPS = 4
SWARM_TOP_K = 6
SWARM_ETA_TOL = 3

# Evacuation
DOOMED_HORIZON = 24
DOOMED_MIN_SHIPS = 8

# Opening
OPENING_TURN = 80
EARLY_TURN = 40
LATE_REMAINING = 65
COMET_MAX_CHASE = 20
COMET_EVAC_HORIZON = 3

# Beam search opening
BEAM_DEPTH = 5
BEAM_WIDTH = 8
BEAM_MAX_WAIT = 15
BEAM_OPENING_LIMIT = 50
BEAM_MAX_PLANETS = 6

# ── geometry & physics ────────────────────────────────────────────────────


def _dist(ax, ay, bx, by):
    return math.hypot(ax - bx, ay - by)


def _fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = min(1.0, max(0.0, math.log(ships) / math.log(1000.0)))
    return 1.0 + (MAX_SPEED - 1.0) * (ratio**1.5)


def _point_to_seg_dist(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-9:
        return _dist(px, py, x1, y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    return _dist(px, py, x1 + t * dx, y1 + t * dy)


def _seg_hits_sun(x1, y1, x2, y2, safety=SUN_SAFE):
    return _point_to_seg_dist(SUN_X, SUN_Y, x1, y1, x2, y2) < SUN_R + safety


def _path_hits_planet(lx, ly, ex, ey, planets, src_id, tgt_id):
    """Check if fleet path hits any planet other than source and target."""
    for p in planets:
        if p.id == src_id or p.id == tgt_id:
            continue
        if _point_to_seg_dist(p.x, p.y, lx, ly, ex, ey) < p.radius:
            return True
    return False


def _launch_pt(sx, sy, sr, angle):
    c = sr + LAUNCH_OFFSET
    return sx + math.cos(angle) * c, sy + math.sin(angle) * c


def _safe_direct(sx, sy, sr, tx, ty, tr):
    """Direct route: returns (angle, distance) or None if sun blocks."""
    angle = math.atan2(ty - sy, tx - sx)
    lx, ly = _launch_pt(sx, sy, sr, angle)
    hit_d = max(0.0, _dist(sx, sy, tx, ty) - (sr + LAUNCH_OFFSET) - tr)
    ex = lx + math.cos(angle) * hit_d
    ey = ly + math.sin(angle) * hit_d
    if _seg_hits_sun(lx, ly, ex, ey):
        return None
    return angle, hit_d


def _waypoint_angle(sx, sy, sr, tx, ty, tr):
    """Try waypoint routing around the sun (from competitive agent idea).

    If direct route is sun-blocked, find a waypoint perpendicular to
    the source-sun line, then aim at that waypoint instead.
    Returns (angle, estimated_total_distance) or None.
    """
    a_src = math.atan2(sy - SUN_Y, sx - SUN_X)
    r = SUN_R + SUN_SAFE + 3.0
    best_wp = None
    best_total = float("inf")

    for offset in (
        math.pi / 2,
        -math.pi / 2,
        math.pi / 3,
        -math.pi / 3,
        2 * math.pi / 3,
        -2 * math.pi / 3,
    ):
        wp_a = a_src + offset
        wx = SUN_X + r * math.cos(wp_a)
        wy = SUN_Y + r * math.sin(wp_a)
        wx = max(1.0, min(99.0, wx))
        wy = max(1.0, min(99.0, wy))

        lx, ly = _launch_pt(sx, sy, sr, math.atan2(wy - sy, wx - sx))
        if _seg_hits_sun(lx, ly, wx, wy):
            continue

        total = _dist(sx, sy, wx, wy) + _dist(wx, wy, tx, ty)
        if total < best_total:
            best_total = total
            best_wp = (wx, wy)

    if best_wp is None:
        return None

    angle = math.atan2(best_wp[1] - sy, best_wp[0] - sx)
    lx, ly = _launch_pt(sx, sy, sr, angle)
    d = _dist(lx, ly, best_wp[0], best_wp[1])
    return angle, d + _dist(best_wp[0], best_wp[1], tx, ty)


def _aim_static(sx, sy, sr, tx, ty, tr):
    """Aim at a static target. Returns (angle, distance) or None."""
    result = _safe_direct(sx, sy, sr, tx, ty, tr)
    if result is not None:
        return result
    return _waypoint_angle(sx, sy, sr, tx, ty, tr)


def _estimate_arrival(sx, sy, sr, tx, ty, tr, ships):
    """Returns (angle, turns) or None if unreachable."""
    aim = _aim_static(sx, sy, sr, tx, ty, tr)
    if aim is None:
        return None
    angle, d = aim
    turns = max(1, int(math.ceil(d / _fleet_speed(max(1, ships)))))
    return angle, turns


def _is_static(planet, initial_by_id=None):
    """Check if planet is static (outer) vs orbiting (inner)."""
    if initial_by_id and planet.id in initial_by_id:
        ip = initial_by_id[planet.id]
        r = _dist(ip.x, ip.y, SUN_X, SUN_Y)
        return r + ip.radius >= ORBIT_LIMIT
    r = _dist(planet.x, planet.y, SUN_X, SUN_Y)
    return r + planet.radius >= ORBIT_LIMIT


def _predict_pos(planet, initial_by_id, ang_vel, turns):
    """Predict planet position at fleet collision time `turns` movement phases from now.

    Fleet collision (phase 2) happens BEFORE orbit advance (phase 3) each step.
    A fleet taking K movement phases arrives at step S+K-1, where the planet
    is at its pre-orbit position = K-1 orbit advances from the observed position.
    """
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    r = _dist(init.x, init.y, SUN_X, SUN_Y)
    if r + init.radius >= ORBIT_LIMIT:
        return planet.x, planet.y
    cur_ang = math.atan2(planet.y - SUN_Y, planet.x - SUN_X)
    new_ang = cur_ang + ang_vel * max(0, turns - 1)
    return SUN_X + r * math.cos(new_ang), SUN_Y + r * math.sin(new_ang)


def _predict_comet_pos(pid, comets, turns):
    """Predict comet position at fleet collision time `turns` movement phases from now.

    Same timing as _predict_pos: collision happens before orbit/comet advance,
    so the comet is at path_index + (turns - 1) at collision time.
    """
    for group in comets:
        pids = group.get("planet_ids", [])
        if pid not in pids:
            continue
        idx = pids.index(pid)
        paths = group.get("paths", [])
        path_index = group.get("path_index", 0)
        if idx >= len(paths):
            return None
        path = paths[idx]
        fi = path_index + max(0, int(turns) - 1)
        if 0 <= fi < len(path):
            return path[fi][0], path[fi][1]
        return None
    return None


def _comet_life(pid, comets):
    for group in comets:
        pids = group.get("planet_ids", [])
        if pid not in pids:
            continue
        idx = pids.index(pid)
        paths = group.get("paths", [])
        path_index = group.get("path_index", 0)
        if idx < len(paths):
            return max(0, len(paths[idx]) - path_index)
    return 0


def _aim_with_prediction(src, target, ships, initial_by_id, ang_vel, comets, comet_ids):
    """Iterative intercept for moving targets. Returns (angle, turns, tx, ty) or None."""
    is_comet = target.id in comet_ids

    # Try direct first
    est = _estimate_arrival(src.x, src.y, src.radius, target.x, target.y, target.radius, ships)
    if est is None:
        if _is_static(target, initial_by_id) and not is_comet:
            return None
        # Search for safe intercept window
        return _search_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids)

    tx, ty = target.x, target.y
    for _ in range(5):
        _, turns = est
        if is_comet:
            pos = _predict_comet_pos(target.id, comets, turns)
        else:
            pos = _predict_pos(target, initial_by_id, ang_vel, turns)
        if pos is None:
            return None
        ntx, nty = pos
        next_est = _estimate_arrival(src.x, src.y, src.radius, ntx, nty, target.radius, ships)
        if next_est is None:
            if _is_static(target, initial_by_id) and not is_comet:
                return None
            return _search_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids)
        if abs(ntx - tx) < 0.3 and abs(nty - ty) < 0.3 and abs(next_est[1] - turns) <= 1:
            # For comets, validate the intercept with step-by-step simulation
            if is_comet and not _validate_comet_intercept(
                src, target, next_est[0], next_est[1], ships, comets
            ):
                return _search_intercept(
                    src, target, ships, initial_by_id, ang_vel, comets, comet_ids
                )
            return next_est[0], next_est[1], ntx, nty
        tx, ty = ntx, nty
        est = next_est

    final = _estimate_arrival(src.x, src.y, src.radius, tx, ty, target.radius, ships)
    if final is None:
        return _search_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids)
    # For comets, validate even after fallback convergence
    if is_comet and not _validate_comet_intercept(src, target, final[0], final[1], ships, comets):
        return None
    return final[0], final[1], tx, ty


def _search_intercept(src, target, ships, initial_by_id, ang_vel, comets, comet_ids):
    """Brute-force search for intercept window on moving target."""
    best = None
    best_score = None
    max_t = min(HORIZON, 60)
    is_comet = target.id in comet_ids
    if is_comet:
        max_t = min(max_t, max(0, _comet_life(target.id, comets)))

    for ct in range(1, max_t + 1):
        if is_comet:
            pos = _predict_comet_pos(target.id, comets, ct)
        else:
            pos = _predict_pos(target, initial_by_id, ang_vel, ct)
        if pos is None:
            continue
        est = _estimate_arrival(src.x, src.y, src.radius, pos[0], pos[1], target.radius, ships)
        if est is None:
            continue
        _, turns = est
        if abs(turns - ct) > 1:
            continue

        actual_t = max(turns, ct)
        if is_comet:
            apos = _predict_comet_pos(target.id, comets, actual_t)
        else:
            apos = _predict_pos(target, initial_by_id, ang_vel, actual_t)
        if apos is None:
            continue
        confirm = _estimate_arrival(
            src.x, src.y, src.radius, apos[0], apos[1], target.radius, ships
        )
        if confirm is None:
            continue
        delta = abs(confirm[1] - actual_t)
        if delta > 1:
            continue
        # Comets move ~4 units/step with radius 1.0: require exact timing match
        # and validate the fleet path actually intersects the comet
        if is_comet:
            if delta > 0:
                continue
            if not _validate_comet_intercept(src, target, confirm[0], confirm[1], ships, comets):
                continue
        score = (delta, confirm[1], ct)
        if best is None or score < best_score:
            best_score = score
            best = (confirm[0], confirm[1], apos[0], apos[1])

    return best


def _validate_comet_intercept(src, target, angle, turns, ships, comets):
    """Simulate fleet path step-by-step and verify it intersects the comet.

    Comets move ~4 units/step so even small timing errors cause misses.
    Check that the fleet's movement segment at each step actually passes
    within the comet's radius of the comet's predicted position.
    """
    speed = _fleet_speed(max(1, int(ships)))
    lx, ly = _launch_pt(src.x, src.y, src.radius, angle)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    for step in range(1, int(turns) + 2):  # check 1 step beyond predicted arrival
        # Fleet position at start and end of this step's movement
        fx0 = lx + cos_a * speed * (step - 1)
        fy0 = ly + sin_a * speed * (step - 1)
        fx1 = lx + cos_a * speed * step
        fy1 = ly + sin_a * speed * step

        # Comet position during this step's collision check
        cpos = _predict_comet_pos(target.id, comets, step)
        if cpos is None:
            continue
        cx, cy = cpos

        # Check if fleet segment passes within comet radius
        d = _point_to_seg_dist(cx, cy, fx0, fy0, fx1, fy1)
        if d < target.radius:
            return True

    return False


# ── fleet destination detection ───────────────────────────────────────────


def _fleet_target(fleet, planets):
    """Find which planet a fleet will hit. Returns (planet, eta_turns) or (None, None)."""
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    speed = _fleet_speed(fleet.ships)
    best_p = None
    best_t = 1e9

    for p in planets:
        dx = p.x - fleet.x
        dy = p.y - fleet.y
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        r_sq = p.radius * p.radius
        if perp_sq >= r_sq:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, r_sq - perp_sq)))
        t = hit_d / speed
        if t <= HORIZON and t < best_t:
            best_t = t
            best_p = p

    if best_p is None:
        return None, None
    return best_p, int(math.ceil(best_t))


# ── arrival simulation ────────────────────────────────────────────────────


def _resolve_combat(owner, garrison, arrivals):
    """Resolve same-turn arrivals at a planet.

    arrivals: list of (eta, attacker_owner, ships)
    Returns (new_owner, new_garrison).
    """
    by_owner = {}
    for _, att_owner, ships in arrivals:
        by_owner[att_owner] = by_owner.get(att_owner, 0) + ships

    if not by_owner:
        return owner, max(0.0, garrison)

    sorted_p = sorted(by_owner.items(), key=lambda x: -x[1])
    top_owner, top_ships = sorted_p[0]
    second_ships = sorted_p[1][1] if len(sorted_p) > 1 else 0

    if top_ships == second_ships:
        survivor_owner = -1
        survivor_ships = 0
    else:
        survivor_owner = top_owner
        survivor_ships = top_ships - second_ships

    if survivor_ships <= 0:
        return owner, max(0.0, garrison)

    if owner == survivor_owner:
        return owner, garrison + survivor_ships
    garrison -= survivor_ships
    if garrison < 0:
        return survivor_owner, -garrison
    return owner, garrison


def _simulate_timeline(planet, arrivals, player, horizon):
    """Simulate a planet's ownership timeline given arriving fleets.

    Returns dict with owner_at, ships_at, keep_needed, fall_turn, holds_full.
    """
    horizon = max(0, int(math.ceil(horizon)))
    # Normalize arrivals
    events = []
    for turns, owner, ships in arrivals:
        if ships <= 0:
            continue
        eta = max(1, int(math.ceil(turns)))
        if eta > horizon:
            continue
        events.append((eta, owner, int(ships)))
    events.sort()

    by_turn = defaultdict(list)
    for item in events:
        by_turn[item[0]].append(item)

    owner = planet.owner
    garrison = float(planet.ships)
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}
    fall_turn = None
    first_enemy = None

    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production

        group = by_turn.get(turn, [])
        prev_owner = owner
        if group:
            if prev_owner == player and first_enemy is None:
                if any(item[1] not in (-1, player) for item in group):
                    first_enemy = turn
            owner, garrison = _resolve_combat(owner, garrison, group)
            if prev_owner == player and owner != player and fall_turn is None:
                fall_turn = turn

        owner_at[turn] = owner
        ships_at[turn] = max(0.0, garrison)

    # Binary search for minimum keep_needed
    keep_needed = 0
    holds_full = True
    if planet.owner == player:

        def _survives(keep):
            o = planet.owner
            g = float(keep)
            for turn in range(1, horizon + 1):
                if o != -1:
                    g += planet.production
                group = by_turn.get(turn, [])
                if group:
                    o, g = _resolve_combat(o, g, group)
                    if o != player:
                        return False
            return o == player

        if _survives(int(planet.ships)):
            lo, hi = 0, int(planet.ships)
            while lo < hi:
                mid = (lo + hi) // 2
                if _survives(mid):
                    hi = mid
                else:
                    lo = mid + 1
            keep_needed = lo
        else:
            holds_full = False
            keep_needed = int(planet.ships)

    return {
        "owner_at": owner_at,
        "ships_at": ships_at,
        "keep_needed": keep_needed,
        "fall_turn": fall_turn,
        "first_enemy": first_enemy,
        "holds_full": holds_full,
        "horizon": horizon,
    }


def _state_at(timeline, turn):
    turn = max(0, min(int(math.ceil(turn)), timeline["horizon"]))
    owner = timeline["owner_at"].get(turn, timeline["owner_at"][timeline["horizon"]])
    ships = timeline["ships_at"].get(turn, timeline["ships_at"][timeline["horizon"]])
    return owner, max(0.0, ships)


# ── lightweight planet/fleet data ─────────────────────────────────────────


class _P:
    """Planet data."""

    __slots__ = ("id", "owner", "x", "y", "radius", "ships", "production")

    def __init__(self, pid, owner, x, y, radius, ships, production):
        self.id = pid
        self.owner = owner
        self.x = x
        self.y = y
        self.radius = radius
        self.ships = ships
        self.production = production


class _F:
    """Fleet data."""

    __slots__ = ("id", "owner", "x", "y", "angle", "from_planet_id", "ships")

    def __init__(self, fid, owner, x, y, angle, from_planet_id, ships):
        self.id = fid
        self.owner = owner
        self.x = x
        self.y = y
        self.angle = angle
        self.from_planet_id = from_planet_id
        self.ships = ships


def _read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _parse_planet(p):
    if hasattr(p, "production"):
        return _P(p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
    if isinstance(p, dict):
        return _P(p["id"], p["owner"], p["x"], p["y"], p["radius"], p["ships"], p["production"])
    return _P(p[0], p[1], p[2], p[3], p[4], p[5], p[6])


def _parse_fleet(f):
    if hasattr(f, "from_planet_id"):
        return _F(f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships)
    if isinstance(f, dict):
        return _F(f["id"], f["owner"], f["x"], f["y"], f["angle"], f["from_planet_id"], f["ships"])
    return _F(f[0], f[1], f[2], f[3], f[4], f[5], f[6])


# ── world model ──────────────────────────────────────────────────────────


class World:
    """Precomputed game state for the current turn."""

    def __init__(self, player, step, planets, fleets, initial_by_id, ang_vel, comets, comet_ids):
        self.player = player
        self.step = step
        self.planets = planets
        self.fleets = fleets
        self.initial_by_id = initial_by_id
        self.ang_vel = ang_vel
        self.comets = comets
        self.comet_ids = set(comet_ids)

        self.by_id = {p.id: p for p in planets}
        self.my_planets = [p for p in planets if p.owner == player]
        self.enemy_planets = [p for p in planets if p.owner not in (-1, player)]
        self.neutral_planets = [p for p in planets if p.owner == -1]
        self.remaining = max(1, MAX_STEPS - step)

        self.is_early = step < EARLY_TURN
        self.is_opening = step < OPENING_TURN
        self.is_late = self.remaining < LATE_REMAINING

        # Strength tracking
        self.strength = defaultdict(int)
        self.production = defaultdict(int)
        for p in planets:
            if p.owner != -1:
                self.strength[p.owner] += int(p.ships)
                self.production[p.owner] += int(p.production)
        for f in fleets:
            self.strength[f.owner] += int(f.ships)

        self.my_total = self.strength.get(player, 0)
        self.enemy_total = sum(s for o, s in self.strength.items() if o != player)
        self.my_prod = self.production.get(player, 0)
        self.num_players = max(
            2, len(set(p.owner for p in planets if p.owner != -1) | set(f.owner for f in fleets))
        )

        # Arrival ledger
        self.arrivals = {p.id: [] for p in planets}
        for f in fleets:
            target, eta = _fleet_target(f, planets)
            if target is not None:
                self.arrivals[target.id].append((eta, f.owner, int(f.ships)))

        # Timelines
        self.timeline = {}
        for p in planets:
            self.timeline[p.id] = _simulate_timeline(p, self.arrivals[p.id], player, HORIZON)

        # Identify leader (for 4-player games)
        self.leader_id = self._find_leader()

        # Initialize caches early (needed by reaction time computation)
        self._aim_cache = {}
        self._need_cache = {}

        # Domination tracking (more aggressive than hybrid)
        total = max(1, self.my_total + self.enemy_total)
        self.domination = (self.my_total - self.enemy_total) / total
        self.is_behind = self.domination < -0.22
        self.is_ahead = self.domination > 0.12
        self.is_finishing = (
            self.domination > 0.28
            and self.my_prod
            > sum(self.production.get(o, 0) for o in self.strength if o != player) * 1.15
            and step > 80
        )

        # Defense ratios
        self.def_ratio = PROACTIVE_DEFENSE_RATIO  # 0.12
        self.stacked_ratio = STACKED_THREAT_RATIO  # 0.18

        # Indirect value map (proximity-based strategic worth)
        self.indirect = {}
        for p in planets:
            friendly = neutral = enemy = 0.0
            for q in planets:
                if q.id == p.id:
                    continue
                d = _dist(p.x, p.y, q.x, q.y)
                if d < 1:
                    continue
                factor = q.production / (d + 12.0)
                if q.owner == player:
                    friendly += factor
                elif q.owner == -1:
                    neutral += factor
                else:
                    enemy += factor
            self.indirect[p.id] = friendly * 0.35 + neutral * 0.9 + enemy * 1.25

        # Reaction times: how fast can we vs enemy reach each non-owned planet
        self.my_react = {}
        self.enemy_react = {}
        for target in planets:
            if target.owner == player:
                continue
            my_best = 1e9
            for src in self.my_planets[:6]:
                seeded = self.best_aim(src.id, target.id, max(1, int(src.ships)))
                if seeded is not None:
                    my_best = min(my_best, seeded[1][1])
            self.my_react[target.id] = my_best
            en_best = 1e9
            for src in self.enemy_planets[:6]:
                seeded = self.best_aim(src.id, target.id, max(1, int(src.ships)))
                if seeded is not None:
                    en_best = min(en_best, seeded[1][1])
            self.enemy_react[target.id] = en_best

        # (caches already initialized above)

        # Isolation scoring: how far is each enemy planet from its nearest ally
        self.isolation = {}
        for p in planets:
            if p.owner in (-1, player):
                continue
            min_ally_dist = 1e9
            for q in planets:
                if q.id == p.id or q.owner != p.owner:
                    continue
                d = _dist(p.x, p.y, q.x, q.y)
                if d < min_ally_dist:
                    min_ally_dist = d
            self.isolation[p.id] = min_ally_dist

        # Crash exploit detection: find planets where enemy fleets will fight
        self.crash_targets = self._detect_crashes()

        # Counter-attack map: enemy planets that recently launched fleets
        # (identified by enemy fleets with from_planet_id)
        self.weakened_enemies = {}
        for f in fleets:
            if f.owner == player or f.owner == -1:
                continue
            src_p = self.by_id.get(f.from_planet_id)
            if src_p is None or src_p.owner != f.owner:
                continue
            # This enemy planet launched ships — track how many are in transit
            self.weakened_enemies[src_p.id] = self.weakened_enemies.get(src_p.id, 0) + int(f.ships)

    def _detect_crashes(self):
        """Detect planets where enemy fleets from different owners converge.

        In 2-player mode, detects when multiple enemy fleets hit a neutral
        or our planet simultaneously, weakening total enemy presence.
        Returns dict of {planet_id: (crash_turn, expected_surviving_ships)}.
        """
        crashes = {}
        for pid, arrs in self.arrivals.items():
            p = self.by_id[pid]
            # Group enemy arrivals by turn windows
            enemy_arrs = [
                (eta, owner, ships)
                for eta, owner, ships in arrs
                if owner != self.player and owner != -1
            ]
            if len(enemy_arrs) < 2:
                continue

            # Check for multi-owner crashes (4-player)
            if self.num_players >= 4:
                by_owner = defaultdict(int)
                min_eta = min(a[0] for a in enemy_arrs)
                for eta, owner, ships in enemy_arrs:
                    if eta <= min_eta + STACKED_THREAT_WINDOW:
                        by_owner[owner] += ships
                if len(by_owner) >= 2:
                    sorted_owners = sorted(by_owner.values(), reverse=True)
                    surviving = sorted_owners[0] - sorted_owners[1]
                    crash_turn = int(min_eta)
                    if surviving < sorted_owners[0] * 0.7:
                        crashes[pid] = (crash_turn, max(0, surviving))

            # In any player count: detect when enemy fleets hit a neutral
            # planet and deplete each other vs garrison
            if p.owner == -1:
                total_enemy = sum(s for _, o, s in enemy_arrs if o != self.player)
                garrison = p.ships
                # After fighting garrison, enemy is weakened
                if total_enemy > garrison and total_enemy < garrison * 3:
                    min_eta = min(a[0] for a in enemy_arrs)
                    surviving = total_enemy - garrison
                    if pid not in crashes or surviving < crashes[pid][1]:
                        crashes[pid] = (int(min_eta), max(0, int(surviving)))

        return crashes

    def _find_leader(self):
        """Find the enemy player with highest ships + production*10."""
        if self.num_players <= 2:
            # In 2-player, there's only one enemy
            for o in self.strength:
                if o != self.player and o >= 0:
                    return o
            return None
        best_id = None
        best_score = -1
        for o in self.strength:
            if o == self.player or o < 0:
                continue
            score = self.strength[o] + self.production.get(o, 0) * 10
            if score > best_score:
                best_score = score
                best_id = o
        return best_id

    def aim(self, src_id, tgt_id, ships):
        """Cached aim_with_prediction. Returns (angle, turns, tx, ty) or None."""
        key = (src_id, tgt_id, int(ships))
        if key in self._aim_cache:
            return self._aim_cache[key]
        src = self.by_id[src_id]
        tgt = self.by_id[tgt_id]
        result = _aim_with_prediction(
            src, tgt, ships, self.initial_by_id, self.ang_vel, self.comets, self.comet_ids
        )
        # Check if path crosses any intermediate planet or goes out of bounds
        if result is not None:
            angle, turns, tx, ty = result
            lx, ly = _launch_pt(src.x, src.y, src.radius, angle)
            # Use distance to predicted target (not speed*turns which overshoots)
            d_target = _dist(lx, ly, tx, ty)
            ex = lx + math.cos(angle) * d_target
            ey = ly + math.sin(angle) * d_target
            # Reject if fleet would leave the board before reaching target
            if not (0.0 <= ex <= BOARD and 0.0 <= ey <= BOARD) or _path_hits_planet(
                lx, ly, ex, ey, self.planets, src_id, tgt_id
            ):
                result = None
        self._aim_cache[key] = result
        return result

    def probe_candidates(self, src_id, tgt_id, cap, hints=()):
        """Generate ship amounts to try for multi-probe aiming."""
        tgt = self.by_id[tgt_id]
        cap = max(1, int(cap))
        tgt_ships = max(1, int(math.ceil(tgt.ships)))

        vals = set(range(1, min(6, cap) + 1))
        vals.update(
            {
                cap,
                max(1, cap // 2),
                max(1, cap // 3),
                min(cap, MIN_PARTIAL_SHIPS),
                min(cap, tgt_ships + 1),
                min(cap, tgt_ships + 2),
                min(cap, tgt_ships + 4),
                min(cap, tgt_ships + 8),
            }
        )
        for h in hints:
            if h is None:
                continue
            base = max(1, min(cap, int(math.ceil(h))))
            for d in (-2, -1, 0, 1, 2):
                c = base + d
                if 1 <= c <= cap:
                    vals.add(c)
        return sorted(vals)

    def best_aim(self, src_id, tgt_id, cap, hints=(), max_turn=None):
        """Find best (ships, (angle, turns, tx, ty)) from probe candidates."""
        best = None
        best_key = None
        for ships in self.probe_candidates(src_id, tgt_id, cap, hints):
            result = self.aim(src_id, tgt_id, ships)
            if result is None:
                continue
            angle, turns, tx, ty = result
            if max_turn is not None and turns > max_turn:
                continue
            key = (turns, ships)
            if best_key is None or key < best_key:
                best_key = key
                best = (ships, (angle, turns, tx, ty))
        return best

    def projected_state(self, tgt_id, arrival_turn, commitments=None, extra=()):
        """Get (owner, ships) at arrival_turn considering commitments."""
        cutoff = max(1, int(math.ceil(arrival_turn)))
        if not (commitments or {}).get(tgt_id) and not extra:
            return _state_at(self.timeline[tgt_id], cutoff)
        all_arr = [a for a in self.arrivals.get(tgt_id, []) if a[0] <= cutoff]
        all_arr.extend(a for a in (commitments or {}).get(tgt_id, []) if a[0] <= cutoff)
        all_arr.extend(a for a in extra if a[0] <= cutoff)
        tgt = self.by_id[tgt_id]
        tl = _simulate_timeline(tgt, all_arr, self.player, cutoff)
        return _state_at(tl, cutoff)

    def projected_timeline(self, tgt_id, horizon, commitments=None, extra=()):
        horizon = max(1, int(math.ceil(horizon)))
        all_arr = [a for a in self.arrivals.get(tgt_id, []) if a[0] <= horizon]
        all_arr.extend(a for a in (commitments or {}).get(tgt_id, []) if a[0] <= horizon)
        all_arr.extend(a for a in extra if a[0] <= horizon)
        tgt = self.by_id[tgt_id]
        return _simulate_timeline(tgt, all_arr, self.player, horizon)

    def ships_to_own(self, tgt_id, eval_turn, commitments=None, extra=(), upper=None):
        """Binary search for minimum ships needed to own target at eval_turn."""
        eval_turn = max(1, int(math.ceil(eval_turn)))

        # Check if already owned
        owner, ships = self.projected_state(tgt_id, eval_turn, commitments, extra)
        if owner == self.player:
            return 0

        def _owns(n):
            o, _ = self.projected_state(
                tgt_id, eval_turn, commitments, tuple(extra) + ((eval_turn, self.player, int(n)),)
            )
            return o == self.player

        if upper is not None:
            hi = max(1, int(upper))
            if not _owns(hi):
                return hi + 1
        else:
            hi = max(1, int(math.ceil(ships)) + 1)
            cap = max(32, sum(int(p.ships) for p in self.planets) + 200)
            while hi <= cap and not _owns(hi):
                hi *= 2
            if hi > cap:
                return cap + 1

        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if _owns(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def reinforce_needed(self, pid, arrival_turn, hold_until, commitments=None, upper=None):
        """How many ships to reinforce planet to hold until hold_until."""
        tgt = self.by_id[pid]
        arrival_turn = max(1, int(math.ceil(arrival_turn)))
        hold_until = max(arrival_turn, int(math.ceil(hold_until)))

        if tgt.owner != self.player:
            return self.ships_to_own(pid, hold_until, commitments, upper=upper)

        def _holds(n):
            tl = self.projected_timeline(
                pid, hold_until, commitments, ((arrival_turn, self.player, int(n)),)
            )
            for t in range(arrival_turn, hold_until + 1):
                if tl["owner_at"].get(t) != self.player:
                    return False
            return True

        if upper is not None:
            hi = max(1, int(upper))
            if not _holds(hi):
                return hi + 1
        else:
            hi = 1
            cap = max(32, sum(int(p.ships) for p in self.planets) + 200)
            while hi <= cap and not _holds(hi):
                hi *= 2
            if hi > cap:
                return cap + 1

        lo = 1
        while lo < hi:
            mid = (lo + hi) // 2
            if _holds(mid):
                hi = mid
            else:
                lo = mid + 1
        return lo


# ── planning logic ────────────────────────────────────────────────────────


def _target_value(target, arrival_turns, mission, world):
    """Score a target by production-horizon value with context multipliers."""
    turns_profit = max(1, world.remaining - arrival_turns)
    if target.id in world.comet_ids:
        life = _comet_life(target.id, world.comets)
        turns_profit = max(0, min(turns_profit, life - arrival_turns))
        if turns_profit <= 0:
            return -1.0

    value = target.production * turns_profit

    # Add indirect strategic value (proximity to other valuable planets)
    value += world.indirect.get(target.id, 0) * turns_profit * 0.15

    # Static vs orbiting
    if _is_static(target, world.initial_by_id):
        if target.owner == -1:
            value *= NEUTRAL_VALUE_MULT
        else:
            value *= 1.55  # static hostile
    elif world.is_opening:
        value *= 0.9  # orbiting planets risky in opening

    # Ownership-based multipliers
    if target.owner == -1:
        # Safe neutral (we can reach before enemy)
        my_t = world.my_react.get(target.id, 1e9)
        en_t = world.enemy_react.get(target.id, 1e9)
        if my_t <= en_t - 2:
            value *= 1.2  # safe
        elif abs(my_t - en_t) <= 2:
            value *= 0.7  # contested - risky
        if world.is_early:
            value *= 1.2
    elif target.owner != world.player:
        # Time-dependent aggression: much more hostile in mid-game
        if world.is_opening:
            value *= OPENING_HOSTILE_MULT
        elif len(world.neutral_planets) <= 3:
            # Most neutrals captured — focus on enemy territory
            value *= 2.8
        else:
            value *= HOSTILE_VALUE_MULT
        # Exposed-enemy bonus: detect when enemy planet has low garrison
        expected_min = target.production * 5
        if target.ships < expected_min and expected_min > 0:
            exposure = 1.0 - target.ships / max(1, expected_min)
            value *= 1.0 + EXPOSED_ENEMY_BONUS * exposure
        # Counter-attack bonus: enemy planet has active fleets in transit
        ships_in_transit = world.weakened_enemies.get(target.id, 0)
        if ships_in_transit > target.ships * 0.3:
            value *= 1.0 + min(
                0.25, ships_in_transit / max(1, target.ships + ships_in_transit) * 0.4
            )

    # Mission-specific
    if mission == "snipe":
        value *= SNIPE_BONUS
    elif mission in ("swarm", "tot_swarm"):
        value *= SWARM_BONUS
    elif mission == "reinforce":
        value *= REINFORCE_BONUS
    elif mission == "crash_exploit":
        value *= CRASH_EXPLOIT_BONUS

    if target.id in world.comet_ids:
        value *= COMET_PENALTY

    # Leader targeting (4-player awareness from competitive agent)
    if world.num_players >= 4 and target.owner == world.leader_id:
        value *= LEADER_BONUS

    # Late game: value ships directly + elimination bonus
    if world.is_late:
        value += max(0, target.ships) * 0.65
        if target.owner not in (-1, world.player):
            enemy_str = world.strength.get(target.owner, 0)
            if enemy_str <= 50:
                value += 22.0  # elimination bonus
            elif enemy_str <= 100:
                value += 8.0  # pressure bonus

    # Domination adjustments
    if world.is_finishing and target.owner not in (-1, world.player):
        value *= 1.18
    if world.is_behind and target.owner == -1 and not _is_static(target, world.initial_by_id):
        value *= 0.92  # avoid risky rotating neutrals when behind

    return value


def _preferred_send(target, base_needed, arrival_turns, available, world):
    """Calculate preferred ship count with safety margins."""
    # Domination-based margin multiplier
    margin_mult = 1.0
    if world.is_ahead:
        margin_mult += 0.06
    if world.is_behind:
        margin_mult -= 0.06
    if world.is_finishing:
        margin_mult += 0.06

    send = max(base_needed, int(math.ceil(base_needed * margin_mult)))
    margin = 0
    if target.owner == -1:
        margin += min(
            NEUTRAL_MARGIN_CAP, NEUTRAL_MARGIN_BASE + target.production * NEUTRAL_MARGIN_PROD
        )
    else:
        margin += min(
            HOSTILE_MARGIN_CAP, HOSTILE_MARGIN_BASE + target.production * HOSTILE_MARGIN_PROD
        )
    if _is_static(target, world.initial_by_id):
        margin += STATIC_MARGIN_BONUS
    if world.num_players >= 4:
        margin += 2
    if arrival_turns > 18:
        margin += min(5, arrival_turns // LONG_TRIP_DIVISOR)
    if target.id in world.comet_ids:
        margin = max(0, margin - 6)
    if world.is_finishing and target.owner not in (-1, world.player):
        margin += 2
    return min(available, send + margin)


def _opening_filter(target, arrival_turns, needed, available, world):
    """Return True if this target should be SKIPPED in opening phase."""
    if not world.is_opening or target.owner != -1:
        return False
    if target.id in world.comet_ids:
        return False
    if _is_static(target, world.initial_by_id):
        return False

    # Safe rotating neutrals — more permissive than hybrid
    my_t = world.my_react.get(target.id, 1e9)
    en_t = world.enemy_react.get(target.id, 1e9)
    # Accept prod >= 3 (hybrid requires 4) with safety gap
    if target.production >= 3 and arrival_turns <= 12 and my_t <= en_t - 2:
        return False

    if world.num_players >= 4:
        affordable = needed <= max(MIN_PARTIAL_SHIPS, int(available * 0.62))
        if affordable and arrival_turns <= 10 and en_t - my_t >= 3:
            return False
        return True

    # Accept longer travel for higher-production targets
    if target.production >= 3:
        return arrival_turns > 16
    return arrival_turns > 13 or target.production <= 1


def _settle(
    src,
    target,
    cap,
    send_guess,
    world,
    commitments,
    mission="capture",
    eval_turn_fn=None,
    anchor_turn=None,
    anchor_tol=None,
    max_iter=4,
):
    """Iteratively converge on the right ship count for a mission.

    Returns (angle, turns, eval_turn, need, send) or None.
    """
    if cap < 1:
        return None
    seed = max(1, min(cap, int(send_guess)))
    eval_turn_fn = eval_turn_fn or (lambda t: t)
    tested = {}

    def _eval(send):
        send = max(1, min(cap, int(send)))
        if send in tested:
            return tested[send]
        result = world.aim(src.id, target.id, send)
        if result is None:
            tested[send] = None
            return None
        angle, turns, tx, ty = result
        if anchor_turn is not None and anchor_tol is not None:
            if abs(turns - anchor_turn) > anchor_tol:
                tested[send] = None
                return None
        raw_eval = int(math.ceil(eval_turn_fn(turns)))
        if raw_eval < turns:
            tested[send] = None
            return None
        need = world.ships_to_own(target.id, raw_eval, commitments, upper=cap)
        if need <= 0 or need > cap:
            tested[send] = None
            return None
        if mission in ("snipe", "crash_exploit", "tot_swarm"):
            desired = need
        elif mission == "rescue":
            desired = min(
                cap, max(need, need + DEFENSE_MARGIN_BASE + target.production * DEFENSE_MARGIN_PROD)
            )
        else:
            desired = min(cap, max(need, _preferred_send(target, need, turns, cap, world)))
        out = (angle, turns, raw_eval, need, send, desired)
        tested[send] = out
        return out

    # Try probe candidates near seed
    candidates = sorted(
        world.probe_candidates(src.id, target.id, cap, hints=(seed,)),
        key=lambda s: (abs(s - seed), s),
    )
    current = None
    for s in candidates:
        r = _eval(s)
        if r is None:
            continue
        if anchor_turn is not None and anchor_tol is not None:
            if abs(r[1] - anchor_turn) > anchor_tol:
                continue
        current = s
        break

    if current is None:
        return None

    for _ in range(max_iter):
        r = _eval(current)
        if r is None:
            break
        angle, turns, eval_t, need, actual, desired = r
        if desired == actual:
            if mission == "rescue" and turns > eval_t:
                return None
            return angle, turns, eval_t, need, actual
        nxt = max(1, min(cap, int(desired)))
        if nxt in tested:
            current = nxt
            break
        current = nxt

    # Fall back to best found
    for s, r in sorted(tested.items(), key=lambda x: (abs(x[0] - seed), x[0])):
        if r is None:
            continue
        angle, turns, eval_t, need, actual, _ = r
        if actual < need:
            continue
        if anchor_turn is not None and anchor_tol is not None:
            if abs(turns - anchor_turn) > anchor_tol:
                continue
        if mission == "rescue" and turns > eval_t:
            continue
        return angle, turns, eval_t, need, actual

    return None


def _settle_reinforce(
    src, target, cap, send_guess, world, commitments, hold_until, max_arrival, max_iter=4
):
    """Settle reinforcement plan. Returns (angle, turns, hold_until, need, send) or None."""
    if cap < 1:
        return None
    seed = max(1, min(cap, int(send_guess)))
    tested = {}

    def _eval(send):
        send = max(1, min(cap, int(send)))
        if send in tested:
            return tested[send]
        result = world.aim(src.id, target.id, send)
        if result is None:
            tested[send] = None
            return None
        angle, turns, tx, ty = result
        if turns > max_arrival:
            tested[send] = None
            return None
        need = world.reinforce_needed(target.id, turns, hold_until, commitments, upper=cap)
        if need <= 0 or need > cap:
            tested[send] = None
            return None
        desired = min(cap, need + REINFORCE_SAFETY)
        out = (angle, turns, hold_until, need, send, desired)
        tested[send] = out
        return out

    candidates = sorted(
        world.probe_candidates(src.id, target.id, cap, hints=(seed,)),
        key=lambda s: (abs(s - seed), s),
    )
    current = None
    for s in candidates:
        r = _eval(s)
        if r is not None:
            current = s
            break
    if current is None:
        return None

    for _ in range(max_iter):
        r = _eval(current)
        if r is None:
            break
        angle, turns, hu, need, actual, desired = r
        if desired == actual:
            return angle, turns, hu, need, actual
        nxt = max(1, min(cap, int(desired)))
        if nxt in tested:
            current = nxt
            break
        current = nxt

    for s, r in sorted(tested.items(), key=lambda x: (abs(x[0] - seed), x[0])):
        if r is None:
            continue
        angle, turns, hu, need, actual, _ = r
        if actual < need or turns > max_arrival:
            continue
        return angle, turns, hu, need, actual
    return None


# ── opening beam search ──────────────────────────────────────────────────


def _beam_best_launch(src_id, src_planet, ref_ships, ref_prod, ref_time, target, world, R):
    """Find optimal launch timing for (src → target). Returns dict or None."""
    G = int(target.ships)
    if ref_prod <= 0 and ref_ships < G + 1:
        return None

    if ref_ships >= G + 1:
        t_min = ref_time
    else:
        need = G + 1 - ref_ships
        t_min = ref_time + int(math.ceil(need / max(1, ref_prod)))

    best = None
    src_static = _is_static(src_planet, world.initial_by_id)
    tgt_static = _is_static(target, world.initial_by_id)

    for extra in range(0, BEAM_MAX_WAIT + 1):
        t = t_min + extra
        if t >= R:
            break
        fleet = ref_ships + ref_prod * (t - ref_time)
        if fleet < G + 1:
            continue
        speed = _fleet_speed(fleet)

        if src_static or t == 0:
            sx, sy = src_planet.x, src_planet.y
        else:
            # Source position at future launch time t: observation-time semantics,
            # not fleet-collision, so add 1 to compensate for _predict_pos's -1.
            sx, sy = _predict_pos(src_planet, world.initial_by_id, world.ang_vel, t + 1)

        if tgt_static:
            eta = _dist(sx, sy, target.x, target.y) / speed
        else:
            eta = _dist(sx, sy, target.x, target.y) / speed
            for _ in range(8):
                px, py = _predict_pos(target, world.initial_by_id, world.ang_vel, t + eta)
                new_eta = _dist(sx, sy, px, py) / speed
                if abs(new_eta - eta) < 0.05:
                    eta = new_eta
                    break
                eta = new_eta

        cap_t = t + eta
        if cap_t >= R:
            continue
        if best is None or cap_t < best["cap_t"]:
            best = {"t_launch": t, "fleet": int(fleet), "eta": eta, "cap_t": cap_t}
        if extra > 5 and best is not None and cap_t > best["cap_t"] + 1.0:
            break

    return best


def _beam_eval_plan(plan, world):
    """Evaluate a beam search plan. Returns {"V": float, "moves": list} or None."""
    R = world.remaining
    sources = {}
    for p in world.my_planets:
        sources[p.id] = (int(p.ships), int(p.production), 0)

    # In-flight captures
    in_flight = set()
    for pid, arrs in world.arrivals.items():
        p = world.by_id.get(pid)
        if p is None or p.owner == world.player:
            continue
        friendly = sorted(
            [(eta, ships) for eta, owner, ships in arrs if owner == world.player],
            key=lambda x: x[0],
        )
        if not friendly:
            continue
        garrison = int(p.ships)
        cum = 0
        for eta, ships in friendly:
            cum += ships
            if cum > garrison:
                sources[pid] = (cum - garrison, int(p.production), eta)
                in_flight.add(pid)
                break

    V = 0.0
    moves = []
    for src_id, tgt_id in plan:
        if src_id not in sources or tgt_id == src_id:
            return None
        ref_ships, ref_prod, ref_t = sources[src_id]
        src_planet = world.by_id[src_id]
        target = world.by_id[tgt_id]
        captured_in_plan = {t for _, t in plan[: len(moves)]}
        if target.owner == world.player and tgt_id not in captured_in_plan:
            return None
        if tgt_id in in_flight and tgt_id not in captured_in_plan:
            return None

        launch = _beam_best_launch(src_id, src_planet, ref_ships, ref_prod, ref_t, target, world, R)
        if launch is None:
            return None

        V += int(target.production) ** 1.3 * (R - launch["cap_t"])
        moves.append(
            {
                "src_id": src_id,
                "tgt_id": tgt_id,
                "t_launch": launch["t_launch"],
                "fleet": launch["fleet"],
                "eta": launch["eta"],
                "cap_t": launch["cap_t"],
                "production": int(target.production),
            }
        )
        sources[src_id] = (0, ref_prod, launch["t_launch"])
        residual = max(0, launch["fleet"] - int(target.ships))
        sources[tgt_id] = (residual, int(target.production), launch["cap_t"])

    return {"V": V, "moves": moves}


def _beam_search(world, depth=BEAM_DEPTH, width=BEAM_WIDTH, deadline=None):
    """Beam search over opening plans."""
    all_targets = [p for p in world.planets if p.owner != world.player]
    if not all_targets:
        return None

    initial_sources = {p.id for p in world.my_planets}
    plans = [{"plan": [], "V": 0.0, "moves": []}]

    for _ in range(depth):
        if deadline and time.perf_counter() >= deadline:
            break
        new_plans = []
        for entry in plans:
            if deadline and time.perf_counter() >= deadline:
                break
            prev = entry["plan"]
            used = {t for _, t in prev}
            avail = initial_sources | used
            for sid in avail:
                for tgt in all_targets:
                    if tgt.id in used or tgt.id == sid:
                        continue
                    new_plan = prev + [(sid, tgt.id)]
                    res = _beam_eval_plan(new_plan, world)
                    if res is None:
                        continue
                    new_plans.append({"plan": new_plan, "V": res["V"], "moves": res["moves"]})
        if not new_plans:
            break
        seen = {}
        for p in new_plans:
            key = tuple(p["plan"])
            if key not in seen or p["V"] > seen[key]["V"]:
                seen[key] = p
        plans = sorted(seen.values(), key=lambda x: -x["V"])[:width]

    if not plans:
        return None
    return max(plans, key=lambda x: x["V"])


def _beam_opening(world):
    """Opening beam search planner. Returns moves list or None."""
    if world.step >= BEAM_OPENING_LIMIT:
        return None
    if not world.my_planets:
        return None
    if world.num_players >= 4:
        return None
    if len(world.my_planets) > BEAM_MAX_PLANETS:
        return None
    # Check if any of our planets is under threat
    for p in world.my_planets:
        ft = world.timeline[p.id]["fall_turn"]
        if ft is not None and ft < 15:
            return None

    n = len(world.my_planets)
    depth = max(2, 6 - n)
    plan_deadline = time.perf_counter() + 0.8
    best = _beam_search(world, depth=depth, width=BEAM_WIDTH, deadline=plan_deadline)
    if best is None or not best["moves"]:
        return None

    moves = []
    for commit in best["moves"]:
        if commit["t_launch"] > 0:
            continue
        src = world.by_id[commit["src_id"]]
        ships = min(int(commit["fleet"]), int(src.ships))
        if ships <= 0:
            continue
        shot = world.aim(src.id, commit["tgt_id"], ships)
        if shot is None:
            continue
        moves.append([src.id, float(shot[0]), ships])
    return moves


# ── opening expansion planner ────────────────────────────────────────────


def _opening_efficiency(target, arrival_turns, ships_needed, world):
    """Production-efficiency score for opening target ranking.

    Returns efficiency = production^2 / (ships_needed + travel_turns*0.5 + 1),
    adjusted by safety, static, and ownership multipliers.
    """
    prod = target.production
    efficiency = (prod * prod) / (ships_needed + arrival_turns * 0.5 + 1)

    # Safety: reaction time comparison
    my_t = world.my_react.get(target.id, 1e9)
    en_t = world.enemy_react.get(target.id, 1e9)
    if my_t <= en_t - 2:
        efficiency *= 1.3  # safe — we arrive first
    elif abs(my_t - en_t) <= 2:
        efficiency *= 0.6  # contested — risky

    # Static planet bonus (easier to aim at)
    if _is_static(target, world.initial_by_id):
        efficiency *= 1.2

    # Hostile planets slightly discounted (harder to take)
    if target.owner not in (-1, world.player):
        efficiency *= 0.85

    return efficiency


def _opening_expand(world, deadline=None):
    """Opening expansion planner (steps 0-79).

    Uses production-efficiency scoring to prioritize targets, with full
    mission support (rescue, capture, snipe, swarm, followup).
    """

    def _expired():
        return deadline is not None and time.perf_counter() > deadline

    def _time_left():
        return (deadline - time.perf_counter()) if deadline else 1e9

    commitments = defaultdict(list)
    moves = []
    spent = defaultdict(int)

    # ── Compute reserves (same as _plan_moves) ──
    reserve = {}
    budget = {}
    for p in world.my_planets:
        keep = world.timeline[p.id]["keep_needed"]
        proactive = 0
        nearby_enemies = []
        for e in world.enemy_planets:
            seeded = world.best_aim(e.id, p.id, max(1, int(e.ships)))
            if seeded is None:
                continue
            _, aim_data = seeded
            if aim_data[1] > PROACTIVE_DEFENSE_HORIZON:
                continue
            proactive = max(proactive, int(e.ships * world.def_ratio))
            nearby_enemies.append((aim_data[1], int(e.ships)))
        if len(nearby_enemies) >= 2:
            nearby_enemies.sort()
            stacked_ships = 0
            for i, (eta_i, ships_i) in enumerate(nearby_enemies):
                window_ships = ships_i
                for j in range(i + 1, len(nearby_enemies)):
                    if nearby_enemies[j][0] - eta_i <= STACKED_THREAT_WINDOW:
                        window_ships += nearby_enemies[j][1]
                if window_ships > stacked_ships:
                    stacked_ships = window_ships
            proactive = max(proactive, int(stacked_ships * world.stacked_ratio))
        reserve[p.id] = min(int(p.ships), max(keep, proactive))
        budget[p.id] = max(0, int(p.ships) - reserve[p.id])

    def _inv_left(pid):
        return max(0, int(world.by_id[pid].ships) - spent[pid])

    def _atk_left(pid):
        return max(0, budget.get(pid, 0) - spent[pid])

    def _add_move(pid, angle, ships, tgt_id=None):
        send = min(int(ships), _inv_left(pid))
        if send < 1:
            return 0
        # Re-aim for orbiting targets when actual send differs from planned
        if tgt_id is not None and send != int(ships):
            tgt = world.by_id.get(tgt_id)
            if tgt and not _is_static(tgt, world.initial_by_id):
                shot = world.aim(pid, tgt_id, send)
                if shot is None:
                    return 0
                angle = shot[0]
        moves.append([pid, float(angle), int(send), tgt_id])
        spent[pid] += send
        return send

    # ── Phase 1: Rescue falling planets ──
    rescue_missions = []
    for target in world.my_planets:
        if _expired():
            break
        ft = world.timeline[target.id]["fall_turn"]
        if ft is None or ft > DEFENSE_LOOKAHEAD:
            continue
        for src in world.my_planets:
            if src.id == target.id:
                continue
            src_avail = _atk_left(src.id)
            if src_avail < MIN_PARTIAL_SHIPS:
                continue
            seeded = world.best_aim(
                src.id,
                target.id,
                src_avail,
                hints=(target.production + DEFENSE_MARGIN_BASE + 2,),
                max_turn=ft,
            )
            if seeded is None:
                continue
            probe, _ = seeded
            plan = _settle(
                src,
                target,
                src_avail,
                probe,
                world,
                commitments,
                mission="rescue",
                eval_turn_fn=lambda _t, _ft=ft: _ft,
                anchor_turn=ft,
            )
            if plan is None:
                continue
            angle, turns, _, need, send_pref = plan
            saved = max(1, world.remaining - ft)
            value = target.production * saved + max(0, target.ships) * 0.55
            score = value / (send_pref + turns * 0.4 + 1.0)
            rescue_missions.append((score, src.id, target.id, angle, turns, need, send_pref, ft))

    # ── Phase 2: Capture missions with efficiency scoring ──
    capture_missions = []
    partial_by_target = defaultdict(list)

    for src in world.my_planets:
        if _expired():
            break
        src_avail = _atk_left(src.id)
        if src_avail <= 0:
            continue

        for target in world.planets:
            if _expired():
                break
            if target.id == src.id or target.owner == world.player:
                continue

            seeded = world.best_aim(src.id, target.id, src_avail, hints=(int(target.ships) + 1,))
            if seeded is None:
                continue
            _, rough = seeded
            rough_turns = rough[1]

            # Comet time check
            if target.id in world.comet_ids:
                life = _comet_life(target.id, world.comets)
                if rough_turns >= life or rough_turns > COMET_MAX_CHASE:
                    continue

            global_need = world.ships_to_own(target.id, rough_turns, commitments, upper=src_avail)
            if global_need <= 0:
                continue

            # Opening filter: skip bad early-game targets
            if _opening_filter(target, rough_turns, global_need, src_avail, world):
                continue

            # Partial contribution (for swarm)
            partial_cap = min(
                src_avail, _preferred_send(target, global_need, rough_turns, src_avail, world)
            )
            if partial_cap >= MIN_PARTIAL_SHIPS:
                partial_aim = world.best_aim(
                    src.id, target.id, partial_cap, hints=(partial_cap, global_need)
                )
                if partial_aim is not None:
                    _, pa = partial_aim
                    eff = _opening_efficiency(target, pa[1], global_need, world)
                    if eff > 0:
                        partial_by_target[target.id].append(
                            (eff, src.id, pa[0], pa[1], global_need, partial_cap)
                        )

            # Full capture
            if global_need <= src_avail:
                send_guess = _preferred_send(target, global_need, rough_turns, src_avail, world)
                plan = _settle(src, target, src_avail, send_guess, world, commitments)
                if plan is None:
                    continue
                angle, turns, _, need, send_cap = plan
                if send_cap < need:
                    continue

                # Comet re-check
                if target.id in world.comet_ids:
                    life = _comet_life(target.id, world.comets)
                    if turns >= life or turns > COMET_MAX_CHASE:
                        continue

                eff = _opening_efficiency(target, turns, need, world)
                if eff <= 0:
                    continue
                capture_missions.append(
                    (eff, src.id, target.id, angle, turns, need, send_cap, "capture")
                )

            # Snipe: time arrival with enemy fleet
            if target.owner == -1:
                enemy_etas = sorted(
                    {
                        int(math.ceil(eta))
                        for eta, owner, ships in world.arrivals.get(target.id, [])
                        if owner not in (-1, world.player) and ships > 0
                    }
                )
                for eeta in enemy_etas[:2]:
                    seeded2 = world.best_aim(
                        src.id, target.id, src_avail, hints=(int(target.ships) + 1,)
                    )
                    if seeded2 is None:
                        continue
                    probe2, _ = seeded2
                    plan = _settle(
                        src,
                        target,
                        src_avail,
                        probe2,
                        world,
                        commitments,
                        mission="snipe",
                        eval_turn_fn=lambda t, _e=eeta: max(t, _e),
                        anchor_turn=eeta,
                        anchor_tol=1,
                    )
                    if plan is None:
                        continue
                    angle, turns, _, need, send_pref = plan
                    eff = _opening_efficiency(target, max(turns, eeta), need, world)
                    if eff <= 0:
                        continue
                    eff *= SNIPE_BONUS
                    capture_missions.append(
                        (eff, src.id, target.id, angle, turns, need, send_pref, "snipe")
                    )

    # ── Phase 3: Build swarm missions ──
    swarm_missions = []
    for tgt_id, options in partial_by_target.items():
        if len(options) < 2:
            continue
        target = world.by_id[tgt_id]
        top = sorted(options, key=lambda x: -x[0])[:SWARM_TOP_K]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                a = top[i]
                b = top[j]
                if a[1] == b[1]:
                    continue
                if abs(a[3] - b[3]) > SWARM_ETA_TOL:
                    continue
                joint_turn = max(a[3], b[3])
                total_cap = a[5] + b[5]
                need = world.ships_to_own(tgt_id, joint_turn, commitments, upper=total_cap)
                if need <= 0 or total_cap < need:
                    continue
                if a[5] >= need or b[5] >= need:
                    continue
                eff = _opening_efficiency(target, joint_turn, need, world)
                if eff <= 0:
                    continue
                eff *= 0.97
                swarm_missions.append((eff, tgt_id, joint_turn, need, [a, b]))

    # ── Phase 4: Execute all missions by priority ──
    all_missions = []
    for m in rescue_missions:
        all_missions.append(("rescue", m[0], m))
    for m in capture_missions:
        all_missions.append(("capture", m[0], m))
    for m in swarm_missions:
        all_missions.append(("swarm", m[0], m))
    all_missions.sort(key=lambda x: -x[1])

    for mtype, mscore, mdata in all_missions:
        if _expired():
            break

        if mtype == "rescue":
            score, src_id, tgt_id, angle, turns, need, send_pref, anchor = mdata
            left = _atk_left(src_id)
            if left <= 0 or left < need:
                continue
            src = world.by_id[src_id]
            target = world.by_id[tgt_id]
            plan = _settle(
                src,
                target,
                left,
                min(left, send_pref),
                world,
                commitments,
                mission="rescue",
                eval_turn_fn=lambda _t, _a=anchor: _a,
                anchor_turn=anchor,
            )
            if plan is None:
                continue
            angle, turns, _, need, send = plan
            if send < need:
                continue
            actual = _add_move(src_id, angle, send, tgt_id)
            if actual >= need:
                commitments[tgt_id].append((turns, world.player, int(actual)))

        elif mtype == "capture":
            score, src_id, tgt_id, angle, turns, need, send_cap, mission = mdata
            left = _atk_left(src_id)
            if left <= 0:
                continue
            src = world.by_id[src_id]
            target = world.by_id[tgt_id]
            plan = _settle(
                src, target, left, min(left, send_cap), world, commitments, mission=mission
            )
            if plan is None:
                continue
            angle, turns, _, need, send = plan
            if send < need:
                continue
            actual = _add_move(src_id, angle, send, tgt_id)
            if actual >= need:
                commitments[tgt_id].append((turns, world.player, int(actual)))

        elif mtype == "swarm":
            score, tgt_id, joint_turn, need, options = mdata
            target = world.by_id[tgt_id]
            limits = []
            for opt in options:
                left = _atk_left(opt[1])
                limits.append(min(left, opt[5]))
            if min(limits) <= 0:
                continue
            total_avail = sum(limits)
            actual_need = world.ships_to_own(tgt_id, joint_turn, commitments, upper=total_avail)
            if actual_need <= 0 or total_avail < actual_need:
                continue
            remaining = actual_need
            sends = {}
            sorted_opts = sorted(zip(options, limits), key=lambda x: (x[0][3], -x[1]))
            for idx, (opt, lim) in enumerate(sorted_opts):
                other_avail = sum(l for _, l in sorted_opts[idx + 1 :])
                s = min(lim, max(0, remaining - other_avail))
                sends[opt[1]] = s
                remaining -= s
            if remaining > 0:
                continue
            reaimed = []
            for opt, _ in sorted_opts:
                s = sends.get(opt[1], 0)
                if s <= 0:
                    continue
                shot = world.aim(opt[1], tgt_id, s)
                if shot is None:
                    reaimed = []
                    break
                reaimed.append((opt[1], shot[0], shot[1], s))
            if not reaimed:
                continue
            turns_list = [r[2] for r in reaimed]
            if max(turns_list) - min(turns_list) > SWARM_ETA_TOL:
                continue
            act_turn = max(turns_list)
            extra_arr = [(t, world.player, s) for _, _, t, s in reaimed]
            o, _ = world.projected_state(tgt_id, act_turn, commitments, extra_arr)
            if o != world.player:
                continue
            committed = []
            for src_id, angle, turns, send in reaimed:
                actual = _add_move(src_id, angle, send, tgt_id)
                if actual > 0:
                    committed.append((turns, world.player, int(actual)))
            if sum(c[2] for c in committed) >= actual_need:
                commitments[tgt_id].extend(committed)

    # ── Phase 5: Followup captures with remaining budget ──
    if _time_left() > OPTIONAL_PHASE_MIN_TIME:
        for src in world.my_planets:
            if _expired():
                break
            src_left = _atk_left(src.id)
            if src_left < 8:
                continue
            best = None
            for target in world.planets:
                if _expired():
                    break
                if target.id == src.id or target.owner == world.player:
                    continue
                seeded = world.best_aim(src.id, target.id, src_left, hints=(int(target.ships) + 1,))
                if seeded is None:
                    continue
                _, rough = seeded
                est_turns = rough[1]
                need = world.ships_to_own(target.id, est_turns, commitments, upper=src_left)
                if need <= 0 or need > src_left:
                    continue
                if _opening_filter(target, est_turns, need, src_left, world):
                    continue
                send = _preferred_send(target, need, est_turns, src_left, world)
                if send < need:
                    continue
                plan = _settle(src, target, src_left, send, world, commitments)
                if plan is None:
                    continue
                _, turns, _, pneed, fsend = plan
                if fsend < pneed:
                    continue
                eff = _opening_efficiency(target, turns, pneed, world)
                if eff <= 0:
                    continue
                if best is None or eff > best[0]:
                    best = (eff, target, plan)
            if best is None:
                continue
            _, target, plan = best
            angle, turns, _, need, send = plan
            src_left = _atk_left(src.id)
            if need > src_left:
                continue
            plan = _settle(src, target, src_left, min(src_left, send), world, commitments)
            if plan is None:
                continue
            angle, turns, _, need, send = plan
            if send < need:
                continue
            actual = _add_move(src.id, angle, send, target.id)
            if actual >= need:
                commitments[target.id].append((turns, world.player, int(actual)))

    # Finalize: clamp to actual planet ships, re-aim orbiting targets
    final = []
    used = defaultdict(int)
    for src_id, angle, ships, tgt_id in moves:
        p = world.by_id[src_id]
        max_ok = int(p.ships) - used[src_id]
        send = min(int(ships), max_ok)
        if send < 1:
            continue
        # Re-aim for orbiting targets if clamped
        if send != int(ships) and tgt_id is not None:
            tgt = world.by_id.get(tgt_id)
            if tgt and not _is_static(tgt, world.initial_by_id):
                shot = world.aim(src_id, tgt_id, send)
                if shot is None:
                    continue
                angle = shot[0]
        final.append([src_id, float(angle), int(send)])
        used[src_id] += send
    return final


# ── main planner ──────────────────────────────────────────────────────────


def _plan_moves(world, deadline=None):
    """Generate moves for this turn using mission-based planning."""

    def _expired():
        return deadline is not None and time.perf_counter() > deadline

    def _time_left():
        return (deadline - time.perf_counter()) if deadline else 1e9

    commitments = defaultdict(list)
    moves = []
    spent = defaultdict(int)

    # Compute reserves (defense budget)
    reserve = {}
    budget = {}
    for p in world.my_planets:
        keep = world.timeline[p.id]["keep_needed"]
        # Proactive defense: reserve against nearby enemies
        proactive = 0
        stacked_ships = 0
        nearby_enemies = []
        for e in world.enemy_planets:
            seeded = world.best_aim(e.id, p.id, max(1, int(e.ships)))
            if seeded is None:
                continue
            _, aim_data = seeded
            if aim_data[1] > PROACTIVE_DEFENSE_HORIZON:
                continue
            proactive = max(proactive, int(e.ships * world.def_ratio))
            nearby_enemies.append((aim_data[1], int(e.ships)))
        # Stacked threat: multiple enemies arriving in a tight window
        if len(nearby_enemies) >= 2:
            nearby_enemies.sort()
            for i, (eta_i, ships_i) in enumerate(nearby_enemies):
                window_ships = ships_i
                for j in range(i + 1, len(nearby_enemies)):
                    if nearby_enemies[j][0] - eta_i <= STACKED_THREAT_WINDOW:
                        window_ships += nearby_enemies[j][1]
                if window_ships > stacked_ships:
                    stacked_ships = window_ships
            stacked_reserve = int(stacked_ships * world.stacked_ratio)
            proactive = max(proactive, stacked_reserve)
        reserve[p.id] = min(int(p.ships), max(keep, proactive))
        budget[p.id] = max(0, int(p.ships) - reserve[p.id])

    def _inv_left(pid):
        return max(0, int(world.by_id[pid].ships) - spent[pid])

    def _atk_left(pid):
        return max(0, budget.get(pid, 0) - spent[pid])

    def _add_move(pid, angle, ships, tgt_id=None):
        send = min(int(ships), _inv_left(pid))
        if send < 1:
            return 0
        # Re-aim for orbiting targets when actual send differs from planned
        if tgt_id is not None and send != int(ships):
            tgt = world.by_id.get(tgt_id)
            if tgt and not _is_static(tgt, world.initial_by_id):
                shot = world.aim(pid, tgt_id, send)
                if shot is None:
                    return 0
                angle = shot[0]
        moves.append([pid, float(angle), int(send), tgt_id])
        spent[pid] += send
        return send

    # ── Phase 1: Rescue missions (defend falling planets) ────────────────
    rescue_missions = []
    for target in world.my_planets:
        ft = world.timeline[target.id]["fall_turn"]
        if ft is None or ft > DEFENSE_LOOKAHEAD:
            continue
        for src in world.my_planets:
            if src.id == target.id:
                continue
            src_avail = _atk_left(src.id)
            if src_avail < MIN_PARTIAL_SHIPS:
                continue
            seeded = world.best_aim(
                src.id,
                target.id,
                src_avail,
                hints=(target.production + DEFENSE_MARGIN_BASE + 2,),
                max_turn=ft,
            )
            if seeded is None:
                continue
            probe, _ = seeded
            plan = _settle(
                src,
                target,
                src_avail,
                probe,
                world,
                commitments,
                mission="rescue",
                eval_turn_fn=lambda _t, _ft=ft: _ft,
                anchor_turn=ft,
            )
            if plan is None:
                continue
            angle, turns, _, need, send_pref = plan
            saved = max(1, world.remaining - ft)
            value = target.production * saved + max(0, target.ships) * 0.55
            if world.enemy_planets:
                nd = min(_dist(target.x, target.y, e.x, e.y) for e in world.enemy_planets)
                if nd < 22:
                    value *= 1.12
            score = value / (send_pref + turns * 0.4 + 1.0)
            rescue_missions.append((score, src.id, target.id, angle, turns, need, send_pref, ft))

    # ── Phase 2: Reinforce missions ──────────────────────────────────────
    reinforce_missions = []
    if world.remaining >= REINFORCE_MIN_FUTURE and _time_left() > HEAVY_PHASE_MIN_TIME:
        for target in world.my_planets:
            ft = world.timeline[target.id]["fall_turn"]
            if ft is None or target.production < REINFORCE_MIN_PROD:
                continue
            hold_until = min(HORIZON, ft + REINFORCE_HOLD_LOOKAHEAD)
            max_arr = min(ft, REINFORCE_MAX_TRAVEL)
            for src in world.my_planets:
                if src.id == target.id:
                    continue
                b = _atk_left(src.id)
                src_cap = min(b, int(src.ships * REINFORCE_MAX_SRC_FRAC))
                if src_cap < MIN_PARTIAL_SHIPS:
                    continue
                seeded = world.best_aim(
                    src.id,
                    target.id,
                    src_cap,
                    hints=(target.production + REINFORCE_SAFETY + 2,),
                    max_turn=max_arr,
                )
                if seeded is None:
                    continue
                probe, _ = seeded
                plan = _settle_reinforce(
                    src, target, src_cap, probe, world, commitments, hold_until, max_arr
                )
                if plan is None:
                    continue
                angle, turns, hu, need, send_pref = plan
                saved = max(1, world.remaining - hu)
                value = target.production * saved * REINFORCE_BONUS
                score = value / (send_pref + turns * 0.35 + 1.0)
                reinforce_missions.append(
                    (score, src.id, target.id, angle, turns, need, send_pref, hu)
                )

    # ── Phase 3: Capture + Snipe missions ────────────────────────────────
    capture_missions = []
    partial_by_target = defaultdict(list)

    for src in world.my_planets:
        if _expired():
            break
        src_avail = _atk_left(src.id)
        if src_avail <= 0:
            continue

        for target in world.planets:
            if _expired():
                break
            if target.id == src.id or target.owner == world.player:
                continue

            seeded = world.best_aim(src.id, target.id, src_avail, hints=(int(target.ships) + 1,))
            if seeded is None:
                continue
            _, rough = seeded
            rough_turns = rough[1]

            # Time validity
            if target.id in world.comet_ids:
                life = _comet_life(target.id, world.comets)
                if rough_turns >= life or rough_turns > COMET_MAX_CHASE:
                    continue
            if world.is_late and rough_turns > world.remaining - 5:
                continue

            global_need = world.ships_to_own(target.id, rough_turns, commitments, upper=src_avail)
            if global_need <= 0:
                continue

            # Opening filter: skip bad early-game targets
            if _opening_filter(target, rough_turns, global_need, src_avail, world):
                continue

            # Partial contribution (for swarm)
            partial_cap = min(
                src_avail, _preferred_send(target, global_need, rough_turns, src_avail, world)
            )
            if partial_cap >= MIN_PARTIAL_SHIPS:
                partial_aim = world.best_aim(
                    src.id, target.id, partial_cap, hints=(partial_cap, global_need)
                )
                if partial_aim is not None:
                    _, pa = partial_aim
                    value = _target_value(target, pa[1], "swarm", world)
                    if value > 0:
                        score = value / (partial_cap + pa[1] * 0.55 + 1.0)
                        partial_by_target[target.id].append(
                            (score, src.id, pa[0], pa[1], global_need, partial_cap)
                        )

            # Full capture
            if global_need <= src_avail:
                send_guess = _preferred_send(target, global_need, rough_turns, src_avail, world)
                plan = _settle(src, target, src_avail, send_guess, world, commitments)
                if plan is None:
                    continue
                angle, turns, _, need, send_cap = plan
                if send_cap < need:
                    continue

                # Time check again
                if target.id in world.comet_ids:
                    life = _comet_life(target.id, world.comets)
                    if turns >= life or turns > COMET_MAX_CHASE:
                        continue
                if world.is_late and turns > world.remaining - 5:
                    continue

                value = _target_value(target, turns, "capture", world)
                if value <= 0:
                    continue
                score = value / (send_cap + turns * 0.55 + 1.0)
                if _is_static(target, world.initial_by_id):
                    score *= STATIC_BONUS
                capture_missions.append(
                    (score, src.id, target.id, angle, turns, need, send_cap, "capture")
                )

            # Snipe: time arrival with enemy fleet
            if target.owner == -1:
                enemy_etas = sorted(
                    {
                        int(math.ceil(eta))
                        for eta, owner, ships in world.arrivals.get(target.id, [])
                        if owner not in (-1, world.player) and ships > 0
                    }
                )
                for eeta in enemy_etas[:2]:
                    seeded2 = world.best_aim(
                        src.id, target.id, src_avail, hints=(int(target.ships) + 1,)
                    )
                    if seeded2 is None:
                        continue
                    probe2, _ = seeded2
                    plan = _settle(
                        src,
                        target,
                        src_avail,
                        probe2,
                        world,
                        commitments,
                        mission="snipe",
                        eval_turn_fn=lambda t, _e=eeta: max(t, _e),
                        anchor_turn=eeta,
                        anchor_tol=1,
                    )
                    if plan is None:
                        continue
                    angle, turns, _, need, send_pref = plan
                    value = _target_value(target, max(turns, eeta), "snipe", world)
                    if value <= 0:
                        continue
                    score = value / (send_pref + max(turns, eeta) * 0.45 + 1.0)
                    score *= SNIPE_BONUS
                    capture_missions.append(
                        (score, src.id, target.id, angle, turns, need, send_pref, "snipe")
                    )

    # ── Phase 3b: Crash exploit missions (strike after enemy fleets fight) ──
    crash_missions = []
    if _time_left() > HEAVY_PHASE_MIN_TIME:
        for pid, (crash_turn, surviving) in world.crash_targets.items():
            if _expired():
                break
            target = world.by_id.get(pid)
            if target is None or target.owner == world.player:
                continue
            # After the crash, the planet will be weakened
            eval_turn = crash_turn + 1
            if eval_turn > world.remaining - 3:
                continue
            for src in world.my_planets:
                src_avail = _atk_left(src.id)
                if src_avail < MIN_PARTIAL_SHIPS:
                    continue
                seeded = world.best_aim(src.id, pid, src_avail, hints=(surviving + 2,))
                if seeded is None:
                    continue
                _, rough = seeded
                if rough[1] < crash_turn:
                    continue  # would arrive before the crash
                plan = _settle(
                    src,
                    target,
                    src_avail,
                    min(src_avail, max(1, surviving + 2)),
                    world,
                    commitments,
                    mission="crash_exploit",
                    eval_turn_fn=lambda t, _ct=crash_turn: max(t, _ct + 1),
                )
                if plan is None:
                    continue
                angle, turns, _, need, send_pref = plan
                if send_pref < need:
                    continue
                value = _target_value(target, turns, "crash_exploit", world)
                if value <= 0:
                    continue
                score = value / (send_pref + turns * 0.45 + 1.0)
                crash_missions.append(
                    (score, src.id, pid, angle, turns, need, send_pref, "crash_exploit")
                )

    # ── Phase 3c: Recapture missions (take back planets that will fall) ──
    recapture_missions = []
    if _time_left() > HEAVY_PHASE_MIN_TIME:
        for target in world.my_planets:
            if _expired():
                break
            tl = world.timeline[target.id]
            ft = tl["fall_turn"]
            if ft is None or tl["holds_full"]:
                continue
            # This planet will fall - plan to recapture it after
            recapture_turn = ft + 2  # give time for enemy to settle
            if recapture_turn > world.remaining - 5:
                continue
            for src in world.my_planets:
                if src.id == target.id:
                    continue
                src_avail = _atk_left(src.id)
                if src_avail < MIN_PARTIAL_SHIPS:
                    continue
                seeded = world.best_aim(
                    src.id, target.id, src_avail, hints=(int(target.ships) + 4,)
                )
                if seeded is None:
                    continue
                _, rough = seeded
                if rough[1] < recapture_turn:
                    continue  # too early, planet hasn't fallen yet
                plan = _settle(
                    src,
                    target,
                    src_avail,
                    min(src_avail, int(target.ships) + 4),
                    world,
                    commitments,
                    eval_turn_fn=lambda t, _rt=recapture_turn: max(t, _rt),
                )
                if plan is None:
                    continue
                angle, turns, _, need, send_pref = plan
                if send_pref < need:
                    continue
                saved = max(1, world.remaining - turns)
                value = target.production * saved * 1.1
                score = value / (send_pref + turns * 0.5 + 1.0)
                recapture_missions.append(
                    (score, src.id, target.id, angle, turns, need, send_pref, turns)
                )

    # ── Phase 4: Build swarm missions from partial contributions ─────────
    swarm_missions = []
    for tgt_id, options in partial_by_target.items():
        if len(options) < 2:
            continue
        target = world.by_id[tgt_id]
        top = sorted(options, key=lambda x: -x[0])[:SWARM_TOP_K]
        # 2-source swarms
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                a = top[i]
                b = top[j]
                if a[1] == b[1]:  # same source
                    continue
                if abs(a[3] - b[3]) > SWARM_ETA_TOL:
                    continue
                joint_turn = max(a[3], b[3])
                total_cap = a[5] + b[5]
                need = world.ships_to_own(tgt_id, joint_turn, commitments, upper=total_cap)
                if need <= 0 or total_cap < need:
                    continue
                if a[5] >= need or b[5] >= need:
                    continue  # single source suffices
                value = _target_value(target, joint_turn, "swarm", world)
                if value <= 0:
                    continue
                score = value / (need + joint_turn * 0.55 + 1.0)
                score *= 0.97  # slight penalty for coordination risk
                swarm_missions.append((score, tgt_id, joint_turn, need, [a, b]))

        # 3-source swarms: for well-defended targets that 2-source can't handle
        if len(top) >= 3:
            for i in range(len(top)):
                for j in range(i + 1, len(top)):
                    for k in range(j + 1, len(top)):
                        a, b, c = top[i], top[j], top[k]
                        srcs = {a[1], b[1], c[1]}
                        if len(srcs) < 3:
                            continue  # need distinct sources
                        etas = [a[3], b[3], c[3]]
                        if max(etas) - min(etas) > SWARM_ETA_TOL:
                            continue
                        joint_turn = max(etas)
                        total_cap = a[5] + b[5] + c[5]
                        need = world.ships_to_own(tgt_id, joint_turn, commitments, upper=total_cap)
                        if need <= 0 or total_cap < need:
                            continue
                        # Must actually need 3 sources (no 2 suffice)
                        if a[5] + b[5] >= need or a[5] + c[5] >= need or b[5] + c[5] >= need:
                            continue
                        value = _target_value(target, joint_turn, "swarm", world)
                        if value <= 0:
                            continue
                        score = value / (need + joint_turn * 0.55 + 1.0)
                        score *= 0.94  # coordination risk penalty for 3-source
                        swarm_missions.append((score, tgt_id, joint_turn, need, [a, b, c]))

    # ── Phase 5: Execute missions by priority ────────────────────────────
    all_missions = []
    for m in rescue_missions:
        all_missions.append(("rescue", m[0], m))
    for m in reinforce_missions:
        all_missions.append(("reinforce", m[0], m))
    for m in capture_missions:
        all_missions.append(("capture", m[0], m))
    for m in crash_missions:
        all_missions.append(("capture", m[0], m))
    for m in recapture_missions:
        all_missions.append(("recapture", m[0], m))
    for m in swarm_missions:
        all_missions.append(("swarm", m[0], m))

    all_missions.sort(key=lambda x: -x[1])

    for mtype, mscore, mdata in all_missions:
        if _expired():
            break

        if mtype == "recapture":
            score, src_id, tgt_id, angle, turns, need, send_pref, anchor = mdata
            left = _atk_left(src_id)
            if left <= 0 or left < need:
                continue
            src = world.by_id[src_id]
            target = world.by_id[tgt_id]
            plan = _settle(src, target, left, min(left, send_pref), world, commitments)
            if plan is None:
                continue
            angle, turns, _, need, send = plan
            if send < need:
                continue
            actual = _add_move(src_id, angle, send, tgt_id)
            if actual >= need:
                commitments[tgt_id].append((turns, world.player, int(actual)))
            continue

        if mtype in ("rescue", "reinforce"):
            score, src_id, tgt_id, angle, turns, need, send_pref, anchor = mdata
            if mtype == "reinforce":
                left = min(
                    _inv_left(src_id), int(world.by_id[src_id].ships * REINFORCE_MAX_SRC_FRAC)
                )
            else:
                left = _atk_left(src_id)
            if left <= 0 or left < need:
                continue

            src = world.by_id[src_id]
            target = world.by_id[tgt_id]
            if mtype == "reinforce":
                plan = _settle_reinforce(
                    src, target, left, min(left, send_pref), world, commitments, anchor, turns
                )
            else:
                plan = _settle(
                    src,
                    target,
                    left,
                    min(left, send_pref),
                    world,
                    commitments,
                    mission="rescue",
                    eval_turn_fn=lambda _t, _a=anchor: _a,
                    anchor_turn=anchor,
                )
            if plan is None:
                continue
            angle, turns, _, need, send = plan
            if send < need:
                continue
            actual = _add_move(src_id, angle, send, tgt_id)
            if actual >= need:
                commitments[tgt_id].append((turns, world.player, int(actual)))

        elif mtype == "capture":
            score, src_id, tgt_id, angle, turns, need, send_cap, mission = mdata
            left = _atk_left(src_id)
            if left <= 0:
                continue
            src = world.by_id[src_id]
            target = world.by_id[tgt_id]
            plan = _settle(
                src, target, left, min(left, send_cap), world, commitments, mission=mission
            )
            if plan is None:
                continue
            angle, turns, _, need, send = plan
            if send < need:
                continue
            actual = _add_move(src_id, angle, send, tgt_id)
            if actual >= need:
                commitments[tgt_id].append((turns, world.player, int(actual)))

        elif mtype == "swarm":
            score, tgt_id, joint_turn, need, options = mdata
            target = world.by_id[tgt_id]
            limits = []
            for opt in options:
                left = _atk_left(opt[1])
                limits.append(min(left, opt[5]))
            if min(limits) <= 0:
                continue

            total_avail = sum(limits)
            actual_need = world.ships_to_own(tgt_id, joint_turn, commitments, upper=total_avail)
            if actual_need <= 0 or total_avail < actual_need:
                continue

            # Distribute ships
            remaining = actual_need
            sends = {}
            sorted_opts = sorted(zip(options, limits), key=lambda x: (x[0][3], -x[1]))
            for idx, (opt, lim) in enumerate(sorted_opts):
                other_avail = sum(l for _, l in sorted_opts[idx + 1 :])
                s = min(lim, max(0, remaining - other_avail))
                sends[opt[1]] = s
                remaining -= s
            if remaining > 0:
                continue

            # Re-aim with actual send amounts
            reaimed = []
            for opt, _ in sorted_opts:
                s = sends.get(opt[1], 0)
                if s <= 0:
                    continue
                shot = world.aim(opt[1], tgt_id, s)
                if shot is None:
                    reaimed = []
                    break
                reaimed.append((opt[1], shot[0], shot[1], s))
            if not reaimed:
                continue

            # Check ETA tolerance
            turns_list = [r[2] for r in reaimed]
            if max(turns_list) - min(turns_list) > SWARM_ETA_TOL:
                continue

            # Verify capture
            act_turn = max(turns_list)
            extra_arr = [(t, world.player, s) for _, _, t, s in reaimed]
            o, _ = world.projected_state(tgt_id, act_turn, commitments, extra_arr)
            if o != world.player:
                continue

            committed = []
            for src_id, angle, turns, send in reaimed:
                actual = _add_move(src_id, angle, send, tgt_id)
                if actual > 0:
                    committed.append((turns, world.player, int(actual)))
            if sum(c[2] for c in committed) >= actual_need:
                commitments[tgt_id].extend(committed)

    # ── Phase 6: Followup captures with remaining budget ─────────────────
    if _time_left() > OPTIONAL_PHASE_MIN_TIME:
        for src in world.my_planets:
            if _expired():
                break
            src_left = _atk_left(src.id)
            if src_left < 8:
                continue

            best = None
            for target in world.planets:
                if _expired():
                    break
                if target.id == src.id or target.owner == world.player:
                    continue
                seeded = world.best_aim(src.id, target.id, src_left, hints=(int(target.ships) + 1,))
                if seeded is None:
                    continue
                _, rough = seeded
                est_turns = rough[1]
                if world.is_late and est_turns > world.remaining - 5:
                    continue

                need = world.ships_to_own(target.id, est_turns, commitments, upper=src_left)
                if need <= 0 or need > src_left:
                    continue
                if _opening_filter(target, est_turns, need, src_left, world):
                    continue

                send = _preferred_send(target, need, est_turns, src_left, world)
                if send < need:
                    continue

                plan = _settle(src, target, src_left, send, world, commitments)
                if plan is None:
                    continue
                _, turns, _, pneed, fsend = plan
                if fsend < pneed:
                    continue

                value = _target_value(target, turns, "capture", world)
                if value <= 0:
                    continue
                score = value / (fsend + turns * 0.55 + 1.0)
                if best is None or score > best[0]:
                    best = (score, target, plan)

            if best is None:
                continue
            _, target, plan = best
            angle, turns, _, need, send = plan
            src_left = _atk_left(src.id)
            if need > src_left:
                continue
            plan = _settle(src, target, src_left, min(src_left, send), world, commitments)
            if plan is None:
                continue
            angle, turns, _, need, send = plan
            if send < need:
                continue
            actual = _add_move(src.id, angle, send, target.id)
            if actual >= need:
                commitments[target.id].append((turns, world.player, int(actual)))

    # ── Phase 7: Evacuate doomed planets ─────────────────────────────────
    if _time_left() > 0.02:
        for p in world.my_planets:
            if _expired():
                break
            status = world.timeline[p.id]
            ft = status["fall_turn"]
            if ft is None or ft > DOOMED_HORIZON:
                continue
            if status["holds_full"]:
                continue
            avail = _inv_left(p.id)
            if avail < DOOMED_MIN_SHIPS:
                continue

            # Try to capture something with the doomed ships
            best_capture = None
            for target in world.planets:
                if target.id == p.id or target.owner == world.player:
                    continue
                seeded = world.best_aim(p.id, target.id, avail, hints=(avail,))
                if seeded is None:
                    continue
                _, pa = seeded
                if pa[1] > world.remaining - 2:
                    continue
                need = world.ships_to_own(target.id, pa[1], commitments, upper=avail)
                if need <= 0 or need > avail:
                    continue
                plan = _settle(
                    p,
                    target,
                    avail,
                    min(avail, max(need, int(target.ships) + 1)),
                    world,
                    commitments,
                )
                if plan is None:
                    continue
                angle, turns, _, pn, send = plan
                if send < pn:
                    continue
                score = _target_value(target, turns, "capture", world) / (send + turns + 1.0)
                if best_capture is None or score > best_capture[0]:
                    best_capture = (score, target.id, angle, turns, send)

            if best_capture is not None:
                _, tgt_id, angle, turns, send = best_capture
                actual = _add_move(p.id, angle, send, tgt_id)
                if actual >= 1:
                    commitments[tgt_id].append((turns, world.player, int(actual)))
                continue

            # Retreat to safe ally
            safe = [
                a for a in world.my_planets if a.id != p.id and world.timeline[a.id]["holds_full"]
            ]
            if not safe:
                continue
            # Pick ally closest to the frontier
            if world.enemy_planets:
                front_d = {
                    a.id: min(_dist(a.x, a.y, e.x, e.y) for e in world.enemy_planets) for a in safe
                }
            else:
                front_d = {a.id: 0 for a in safe}
            retreat = min(safe, key=lambda a: (front_d.get(a.id, 1e9), _dist(p.x, p.y, a.x, a.y)))
            shot = world.aim(p.id, retreat.id, avail)
            if shot is not None:
                _add_move(p.id, shot[0], avail, retreat.id)

    # ── Phase 8: Stage rear planets forward ──────────────────────────────
    if not world.is_late and _time_left() > OPTIONAL_PHASE_MIN_TIME and len(world.my_planets) > 1:
        if world.enemy_planets or world.neutral_planets:
            frontier_targets = world.enemy_planets or world.neutral_planets
            front_d = {
                p.id: min(_dist(p.x, p.y, t.x, t.y) for t in frontier_targets)
                for p in world.my_planets
            }
            safe_fronts = [p for p in world.my_planets if world.timeline[p.id]["holds_full"]]
            if safe_fronts:
                front_anchor = min(safe_fronts, key=lambda p: front_d[p.id])
                send_ratio = 0.7 if world.num_players >= 4 else 0.62

                for rear in sorted(world.my_planets, key=lambda p: -front_d[p.id]):
                    if _expired():
                        break
                    if rear.id == front_anchor.id:
                        continue
                    if _atk_left(rear.id) < 16:
                        continue
                    if front_d[rear.id] < front_d[front_anchor.id] * 1.25:
                        continue

                    # Find staging planet
                    stage = [
                        a
                        for a in safe_fronts
                        if a.id != rear.id and front_d[a.id] < front_d[rear.id] * 0.78
                    ]
                    if stage:
                        front = min(stage, key=lambda a: _dist(rear.x, rear.y, a.x, a.y))
                    else:
                        continue

                    send = int(_atk_left(rear.id) * send_ratio)
                    if send < 10:
                        continue
                    shot = world.aim(rear.id, front.id, send)
                    if shot is None:
                        continue
                    if shot[1] > 40:
                        continue
                    _add_move(rear.id, shot[0], send, front.id)

    # Finalize: clamp to actual planet ships, re-aim orbiting targets
    final = []
    used = defaultdict(int)
    for src_id, angle, ships, tgt_id in moves:
        p = world.by_id[src_id]
        max_ok = int(p.ships) - used[src_id]
        send = min(int(ships), max_ok)
        if send < 1:
            continue
        # Re-aim for orbiting targets if clamped
        if send != int(ships) and tgt_id is not None:
            tgt = world.by_id.get(tgt_id)
            if tgt and not _is_static(tgt, world.initial_by_id):
                shot = world.aim(src_id, tgt_id, send)
                if shot is None:
                    continue
                angle = shot[0]
        final.append([src_id, float(angle), int(send)])
        used[src_id] += send
    return final


# ── comet evacuation ──────────────────────────────────────────────────────


def _evacuate_comets(world, moves):
    """Send all ships from owned comets about to exit the grid to the best target."""
    if not world.comet_ids:
        return moves

    sent = defaultdict(int)
    for m in moves:
        sent[m[0]] += m[2]

    for p in world.my_planets:
        if p.id not in world.comet_ids:
            continue
        life = _comet_life(p.id, world.comets)
        if life > COMET_EVAC_HORIZON or life <= 0:
            continue
        remaining = int(p.ships) - sent.get(p.id, 0)
        if remaining < 1:
            continue

        # Find best reachable non-comet target (prefer friendly, high production)
        best_angle = None
        best_value = -1.0
        for t in world.planets:
            if t.id == p.id or t.id in world.comet_ids:
                continue
            shot = world.aim(p.id, t.id, remaining)
            if shot is None:
                continue
            value = t.production
            if t.owner == world.player:
                value += 10
            if value > best_value:
                best_value = value
                best_angle = shot[0]

        if best_angle is not None:
            moves.append([p.id, float(best_angle), remaining])

    return moves


# ── main agent function (MUST be last callable) ──────────────────────────


def agent(obs, config=None):
    """Apex agent merging hybrid + peaking bot strategies."""
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

    world = World(player, step, planets, fleets, initial_by_id, ang_vel, comets, comet_ids)

    if not world.my_planets:
        return []

    # Try opening beam search first (2-player, ≤6 planets, steps 0-50)
    beam_result = _beam_opening(world)
    if beam_result is not None:
        return beam_result

    # Time budget
    act_timeout = _read(config, "actTimeout", 1.0) if config else 1.0
    soft = min(SOFT_DEADLINE_FRAC, max(0.55, act_timeout * 0.82))
    deadline = start + soft

    # Opening: efficiency-focused planner (steps 0-79, when beam doesn't apply)
    if world.is_opening:
        moves = _opening_expand(world, deadline=deadline)
    else:
        # Mid/late game: mission-based planner
        moves = _plan_moves(world, deadline=deadline)

    # Evacuate owned comets about to exit the grid
    return _evacuate_comets(world, moves)


if __name__ == "__main__":
    from kaggle_environments import make

    env = make("orbit_wars", debug=True)
    env.run([agent, "random"])
    final = env.steps[-1]
    for i, s in enumerate(final):
        print(f"Player {i}: reward={s.reward}, status={s.status}")
