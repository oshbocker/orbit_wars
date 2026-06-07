"""Fleet destination prediction for V2 pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.features import (
    BOARD_SIZE,
    MAX_SHIP_SPEED,
    SUN_RADIUS,
    SUN_SAFE_RADIUS,
    fleet_hits_planet,
    fleet_speed,
    passes_through_sun,
    planet_pos_at,
)
from src.game_types import SUN_X, SUN_Y, FleetState, GameState, PlanetState

_LOG1000 = math.log(1000)
_ORBIT_MAX_STEPS = 100  # mirrors _orbiting_hit_check max_steps


@dataclass(slots=True)
class IncomingFleetInfo:
    """Per-planet incoming fleet aggregation, indexed by relative team (0=own, 1-3=enemies)."""

    ships: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    eta: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])


def predict_fleet_destination(
    fleet: FleetState,
    planets: list[PlanetState],
    step: int,
    angular_velocity: float,
) -> tuple[PlanetState | None, float]:
    """Predict which planet a fleet will hit.

    Static planets: ray-circle intersection via fleet_hits_planet().
    Orbiting planets: step-by-step forward simulation.
    Returns (target_planet, eta_turns) or (None, inf) if no hit.
    """
    best_planet: PlanetState | None = None
    best_eta = float("inf")

    speed = fleet_speed(fleet.ships)

    for planet in planets:
        if not planet.is_orbiting:
            # Static planet: direct ray-circle check
            eta = fleet_hits_planet(fleet, planet)
            if eta is not None and eta < best_eta:
                best_eta = eta
                best_planet = planet
        else:
            # Orbiting planet: step-by-step forward sim
            eta = _orbiting_hit_check(fleet, planet, step, angular_velocity, speed)
            if eta is not None and eta < best_eta:
                best_eta = eta
                best_planet = planet

    if best_planet is None:
        return None, float("inf")

    # Check if fleet hits sun before reaching the planet
    fx, fy = fleet.x, fleet.y
    dx = math.cos(fleet.angle) * speed
    dy = math.sin(fleet.angle) * speed
    hit_x = fx + dx * best_eta
    hit_y = fy + dy * best_eta
    if passes_through_sun(fx, fy, hit_x, hit_y):
        return None, float("inf")

    return best_planet, best_eta


def _orbiting_hit_check(
    fleet: FleetState,
    planet: PlanetState,
    step: int,
    angular_velocity: float,
    speed: float,
    max_steps: int = 100,
) -> float | None:
    """Check if fleet collides with an orbiting planet via forward sim."""
    fx, fy = fleet.x, fleet.y
    dx = math.cos(fleet.angle) * speed
    dy = math.sin(fleet.angle) * speed
    hit_r = planet.radius + 0.5

    for t in range(1, max_steps + 1):
        # Fleet position at time t
        new_fx = fx + dx * t
        new_fy = fy + dy * t

        # Out of bounds check
        if new_fx < 0 or new_fx > BOARD_SIZE or new_fy < 0 or new_fy > BOARD_SIZE:
            return None

        # Planet position during fleet movement (before orbit advance)
        px, py = planet_pos_at(planet, step + t - 1, angular_velocity)

        # Check collision: continuous segment from (prev_fx, prev_fy) to (new_fx, new_fy)
        prev_fx = fx + dx * (t - 1)
        prev_fy = fy + dy * (t - 1)
        if _segment_circle_hit(prev_fx, prev_fy, new_fx, new_fy, px, py, hit_r):
            return float(t)

    return None


def _segment_circle_hit(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    cx: float,
    cy: float,
    r: float,
) -> bool:
    """Check if line segment (x1,y1)→(x2,y2) intersects circle (cx,cy,r)."""
    sdx, sdy = x2 - x1, y2 - y1
    fx, fy = x1 - cx, y1 - cy
    a = sdx * sdx + sdy * sdy
    if a < 1e-10:
        return fx * fx + fy * fy <= r * r
    b = 2 * (fx * sdx + fy * sdy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2 * a)
    t2 = (-b + sq) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1) or (t1 < 0 < t2)


def _compute_incoming_fleets_scalar(
    state: GameState,
    player: int,
    enemy_order: list[int] | None = None,
) -> dict[int, IncomingFleetInfo]:
    """Reference (scalar) implementation, kept for equivalence testing.

    `compute_incoming_fleets` is the vectorized production path; this mirrors it
    line-for-line via the per-fleet `predict_fleet_destination`.
    """
    result: dict[int, IncomingFleetInfo] = {}
    enemy_ids: list[int] = list(enemy_order) if enemy_order else []

    for fleet in state.fleets:
        target, eta = predict_fleet_destination(
            fleet,
            state.planets,
            state.step,
            state.angular_velocity,
        )
        if target is None:
            continue

        # Determine relative team index
        if fleet.owner == player:
            team = 0
        else:
            if fleet.owner not in enemy_ids:
                enemy_ids.append(fleet.owner)
            idx = enemy_ids.index(fleet.owner)
            team = min(idx + 1, 3)  # cap at 3

        if target.id not in result:
            result[target.id] = IncomingFleetInfo()
        info = result[target.id]

        # Weighted average ETA
        old_ships = info.ships[team]
        new_total = old_ships + fleet.ships
        if new_total > 0:
            info.eta[team] = (info.eta[team] * old_ships + eta * fleet.ships) / new_total
        info.ships[team] = new_total

    return result


def _predict_destinations_batch(
    state: GameState,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized port of `predict_fleet_destination` over all fleets at once.

    Returns (best_planet_idx, best_eta) as arrays of length len(state.fleets):
      best_planet_idx[k] = index into state.planets of the planet fleet k hits,
                           or -1 if it hits nothing (or is sun-blocked).
      best_eta[k]        = ETA in turns to that planet (inf if no target).

    The geometry exactly mirrors the scalar helpers: static planets use the
    `fleet_hits_planet` closed form (ray-circle + edge + sun-occlusion), orbiting
    planets use the `_orbiting_hit_check` forward sim (positions precomputed once
    per planet, segment-circle test broadcast across fleets/timesteps), and the
    final pick passes the chosen hit point through `passes_through_sun`.
    """
    fleets = state.fleets
    planets = state.planets
    F = len(fleets)
    Pn = len(planets)
    if F == 0 or Pn == 0:
        return np.full(F, -1, dtype=np.int64), np.full(F, np.inf)

    # ── Fleet arrays ─────────────────────────────────────────────────────────
    fx = np.array([f.x for f in fleets], dtype=np.float64)
    fy = np.array([f.y for f in fleets], dtype=np.float64)
    fang = np.array([f.angle for f in fleets], dtype=np.float64)
    fsh = np.array([f.ships for f in fleets], dtype=np.float64)
    # fleet_speed, vectorized (ships<=1 -> 1.0).
    spd = np.where(
        fsh <= 1.0,
        1.0,
        1.0 + (MAX_SHIP_SPEED - 1.0) * (np.log(np.maximum(fsh, 1.0001)) / _LOG1000) ** 1.5,
    )
    dx = np.cos(fang) * spd
    dy = np.sin(fang) * spd
    a = dx * dx + dy * dy  # [F]; >=1 since spd>=1

    # ── Planet arrays ────────────────────────────────────────────────────────
    px = np.array([p.x for p in planets], dtype=np.float64)
    py = np.array([p.y for p in planets], dtype=np.float64)
    prad = np.array([p.radius for p in planets], dtype=np.float64)
    orbiting = np.array([p.is_orbiting for p in planets], dtype=bool)
    hit_r = prad + 0.5  # [Pn]

    eta_full = np.full((F, Pn), np.inf, dtype=np.float64)

    # ── Static planets: vectorized fleet_hits_planet ──────────────────────────
    stat = ~orbiting
    if stat.any():
        sx = px[stat]
        sy = py[stat]
        sr = hit_r[stat]
        fxp = fx[:, None] - sx[None, :]  # [F, Ns]
        fyp = fy[:, None] - sy[None, :]
        b = 2.0 * (fxp * dx[:, None] + fyp * dy[:, None])
        c = fxp * fxp + fyp * fyp - sr[None, :] ** 2
        disc = b * b - 4.0 * a[:, None] * c
        ok = disc >= 0.0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        twoa = 2.0 * a[:, None]
        t1 = (-b - sq) / twoa
        t2 = (-b + sq) / twoa
        t = np.where(t1 > 0.0, t1, t2)
        ok &= t > 0.0
        # Out-of-bounds hit point allowed only if it occurs before the board edge.
        hx = fx[:, None] + dx[:, None] * t
        hy = fy[:, None] + dy[:, None] * t
        oob = (hx < 0) | (hx > BOARD_SIZE) | (hy < 0) | (hy > BOARD_SIZE)
        inf = np.inf
        e1 = np.where(dx > 0, (BOARD_SIZE - fx) / np.maximum(dx, 1e-10), inf)
        e2 = np.where(dx < 0, -fx / np.minimum(dx, -1e-10), inf)
        e3 = np.where(dy > 0, (BOARD_SIZE - fy) / np.maximum(dy, 1e-10), inf)
        e4 = np.where(dy < 0, -fy / np.minimum(dy, -1e-10), inf)
        d_edge = np.minimum(np.minimum(e1, e2), np.minimum(e3, e4))  # [F]
        ok &= ~(oob & (t > d_edge[:, None]))
        # Sun occlusion (SUN_RADIUS) before reaching the planet.
        sfx = fx - SUN_X
        sfy = fy - SUN_Y
        sun_b = 2.0 * (sfx * dx + sfy * dy)
        sun_c = sfx * sfx + sfy * sfy - SUN_RADIUS * SUN_RADIUS
        sun_disc = sun_b * sun_b - 4.0 * a * sun_c
        sun_ok = sun_disc >= 0.0
        sun_sq = np.sqrt(np.where(sun_ok, sun_disc, 0.0))
        sun_t = (-sun_b - sun_sq) / (2.0 * a)  # [F]
        sun_block = sun_ok[:, None] & (sun_t[:, None] > 0.0) & (sun_t[:, None] < t)
        ok &= ~sun_block
        stat_eta = np.where(ok, t, np.inf)
        eta_full[:, np.where(stat)[0]] = stat_eta

    # ── Orbiting planets: vectorized _orbiting_hit_check forward sim ───────────
    orb_idx = np.where(orbiting)[0]
    if orb_idx.size > 0:
        init_ang = np.array([planets[i].initial_angle for i in orb_idx], dtype=np.float64)
        orb_r = np.array([planets[i].orbital_radius for i in orb_idx], dtype=np.float64)
        orb_hr = hit_r[orb_idx]  # [M]
        tarr = np.arange(1, _ORBIT_MAX_STEPS + 1, dtype=np.float64)  # [T]
        # Planet positions at step + t - 1 (independent of fleet) -> [M, T].
        ang = init_ang[:, None] + state.angular_velocity * (state.step + tarr[None, :] - 1.0)
        cpx = SUN_X + orb_r[:, None] * np.cos(ang)
        cpy = SUN_Y + orb_r[:, None] * np.sin(ang)
        # Fleet prev/new positions -> [F, T].
        fpx = fx[:, None] + dx[:, None] * (tarr[None, :] - 1.0)
        fpy = fy[:, None] + dy[:, None] * (tarr[None, :] - 1.0)
        fnx = fx[:, None] + dx[:, None] * tarr[None, :]
        fny = fy[:, None] + dy[:, None] * tarr[None, :]
        # First timestep where the fleet new-pos leaves the board (loop stops there).
        oob = (fnx < 0) | (fnx > BOARD_SIZE) | (fny < 0) | (fny > BOARD_SIZE)  # [F,T]
        oob_t = np.where(oob, tarr[None, :], np.inf)
        first_oob = oob_t.min(axis=1)  # [F]
        # Segment-circle hit for every (fleet, orbiting-planet, timestep) -> [F,M,T].
        relx = fpx[:, None, :] - cpx[None, :, :]
        rely = fpy[:, None, :] - cpy[None, :, :]
        bb = 2.0 * (relx * dx[:, None, None] + rely * dy[:, None, None])
        cc = relx * relx + rely * rely - (orb_hr[None, :, None] ** 2)
        aa = a[:, None, None]
        dd = bb * bb - 4.0 * aa * cc
        hit_geo = dd >= 0.0
        ssq = np.sqrt(np.where(hit_geo, dd, 0.0))
        ot1 = (-bb - ssq) / (2.0 * aa)
        ot2 = (-bb + ssq) / (2.0 * aa)
        seg = hit_geo & (
            ((ot1 >= 0.0) & (ot1 <= 1.0))
            | ((ot2 >= 0.0) & (ot2 <= 1.0))
            | ((ot1 < 0.0) & (ot2 > 0.0))
        )
        # Only timesteps strictly before the fleet leaves the board count.
        seg &= tarr[None, None, :] < first_oob[:, None, None]
        any_hit = seg.any(axis=2)  # [F,M]
        first_t = seg.argmax(axis=2)  # [F,M] first True idx
        orb_eta = np.where(any_hit, tarr[first_t], np.inf)
        eta_full[:, orb_idx] = orb_eta

    # ── Pick nearest planet (first on ties) + final sun gate ──────────────────
    best_idx = np.argmin(eta_full, axis=1)
    best_eta = eta_full[np.arange(F), best_idx]
    valid = np.isfinite(best_eta)
    # Final sun check on the chosen hit point (passes_through_sun, SUN_SAFE_RADIUS).
    # Use a finite stand-in ETA for invalid fleets so the inf-arithmetic below
    # doesn't emit warnings; those entries are masked out via `valid` regardless.
    safe_eta = np.where(valid, best_eta, 0.0)
    bhx = fx + dx * safe_eta
    bhy = fy + dy * safe_eta
    pdx = bhx - fx
    pdy = bhy - fy
    pa = pdx * pdx + pdy * pdy
    psfx = fx - SUN_X
    psfy = fy - SUN_Y
    pb = 2.0 * (psfx * pdx + psfy * pdy)
    pc = psfx * psfx + psfy * psfy - SUN_SAFE_RADIUS**2
    pdisc = pb * pb - 4.0 * pa * pc
    with np.errstate(invalid="ignore", divide="ignore"):
        psq = np.sqrt(np.where(pdisc >= 0.0, pdisc, 0.0))
        pt1 = (-pb - psq) / (2.0 * pa)
        pt2 = (-pb + psq) / (2.0 * pa)
    sun_hit = (
        (pa > 0.0)
        & (pdisc >= 0.0)
        & valid
        & (
            ((pt1 >= 0.0) & (pt1 <= 1.0))
            | ((pt2 >= 0.0) & (pt2 <= 1.0))
            | ((pt1 < 0.0) & (pt2 > 0.0))
        )
    )
    valid &= ~sun_hit
    best_idx = np.where(valid, best_idx, -1)
    best_eta = np.where(valid, best_eta, np.inf)
    return best_idx.astype(np.int64), best_eta


def compute_incoming_fleets(
    state: GameState,
    player: int,
    enemy_order: list[int] | None = None,
) -> dict[int, IncomingFleetInfo]:
    """For each planet, aggregate all incoming fleets by relative team.

    Team 0 = player's own fleets
    Teams 1-3 = enemies. If `enemy_order` is given (v4 Tier 2.5 stable ordering),
    enemies map to slots by that canonical list so the team indices match the
    planet ownership one-hot; otherwise slots are assigned by first encounter.

    Geometry (which planet each fleet hits + ETA) is computed in one vectorized
    pass (`_predict_destinations_batch`); the team-mapping + weighted-ETA
    aggregation stays a scalar per-fleet loop so it is bit-identical to the
    reference `_compute_incoming_fleets_scalar`.
    """
    result: dict[int, IncomingFleetInfo] = {}
    if not state.fleets:
        return result

    best_idx, best_eta = _predict_destinations_batch(state)
    planets = state.planets
    enemy_ids: list[int] = list(enemy_order) if enemy_order else []

    for k, fleet in enumerate(state.fleets):
        pidx = int(best_idx[k])
        if pidx < 0:
            continue
        target = planets[pidx]
        eta = float(best_eta[k])

        # Determine relative team index
        if fleet.owner == player:
            team = 0
        else:
            if fleet.owner not in enemy_ids:
                enemy_ids.append(fleet.owner)
            idx = enemy_ids.index(fleet.owner)
            team = min(idx + 1, 3)  # cap at 3

        if target.id not in result:
            result[target.id] = IncomingFleetInfo()
        info = result[target.id]

        # Weighted average ETA
        old_ships = info.ships[team]
        new_total = old_ships + fleet.ships
        if new_total > 0:
            info.eta[team] = (info.eta[team] * old_ships + eta * fleet.ships) / new_total
        info.ships[team] = new_total

    return result
