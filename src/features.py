from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import EnvConfig
from .game_types import FleetState, GameState, PlanetState, SUN_X, SUN_Y

# ── constants ────────────────────────────────────────────────────────────────

SUN_RADIUS = 10.0
SUN_SAFE_RADIUS = SUN_RADIUS + 2.0
MAX_SHIP_SPEED = 6.0
BOARD_SIZE = 100.0
MAX_STEPS = 500

# Feature dimensions
GLOBAL_DIM = 9
SOURCE_SCALAR_DIM = 7
KNN_SCALAR_DIM = 4
TARGET_SCALAR_DIM = 11


# ── fleet physics ────────────────────────────────────────────────────────────

def fleet_speed(ships: float) -> float:
    if ships <= 1:
        return 1.0
    return 1.0 + (MAX_SHIP_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5


def passes_through_sun(x1: float, y1: float, x2: float, y2: float) -> bool:
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


def fleet_hits_planet(fleet: FleetState, planet: PlanetState) -> float | None:
    """Return ETA in turns for fleet to hit planet, or None if it misses."""
    speed = fleet_speed(fleet.ships)
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

    # Check sun intersection before planet hit
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


def planet_pos_at(planet: PlanetState, future_step: int, angular_velocity: float) -> tuple[float, float]:
    if not planet.is_orbiting:
        return planet.x, planet.y
    angle = planet.initial_angle + angular_velocity * future_step
    return SUN_X + planet.orbital_radius * math.cos(angle), SUN_Y + planet.orbital_radius * math.sin(angle)


def aim_angle(src: PlanetState, tgt: PlanetState) -> float:
    """Direct angle from source to target (no sun avoidance for simplicity in features)."""
    return math.atan2(tgt.y - src.y, tgt.x - src.x)


def safe_angle(src_x: float, src_y: float, dst_x: float, dst_y: float) -> tuple[float, bool]:
    """Compute angle with sun avoidance. Returns (angle, was_redirected)."""
    direct = math.atan2(dst_y - src_y, dst_x - src_x)
    if not passes_through_sun(src_x, src_y, dst_x, dst_y):
        return direct, False

    a_src = math.atan2(src_y - SUN_Y, src_x - SUN_X)
    r = SUN_SAFE_RADIUS + 3.0
    best_wp = None
    best_total = float("inf")
    for offset in (math.pi / 2, -math.pi / 2, math.pi / 3, -math.pi / 3,
                   2 * math.pi / 3, -2 * math.pi / 3):
        wp_a = a_src + offset
        wx = max(1.0, min(99.0, SUN_X + r * math.cos(wp_a)))
        wy = max(1.0, min(99.0, SUN_Y + r * math.sin(wp_a)))
        if passes_through_sun(src_x, src_y, wx, wy):
            continue
        total = math.hypot(src_x - wx, src_y - wy) + math.hypot(wx - dst_x, wy - dst_y)
        if total < best_total:
            best_total = total
            best_wp = (wx, wy)

    if best_wp is None:
        return direct, False
    return math.atan2(best_wp[1] - src_y, best_wp[0] - src_x), True


# ── fleet transit computation ────────────────────────────────────────────────

@dataclass
class TransitInfo:
    """Per-planet fleet transit aggregation."""
    enemy_ships: float = 0.0
    enemy_eta: float = 0.0
    friendly_ships: float = 0.0
    friendly_eta: float = 0.0


@dataclass
class FleetTransitState:
    """Mutable transit state, updated as sequential decisions are made."""
    transit: dict[int, TransitInfo] = field(default_factory=dict)

    def get(self, planet_id: int) -> TransitInfo:
        return self.transit.get(planet_id, TransitInfo())

    def add_fleet(self, planet_id: int, ships: float, eta: float, is_friendly: bool) -> None:
        if planet_id not in self.transit:
            self.transit[planet_id] = TransitInfo()
        info = self.transit[planet_id]
        if is_friendly:
            total = info.friendly_ships + ships
            if total > 0:
                info.friendly_eta = (info.friendly_eta * info.friendly_ships + eta * ships) / total
            info.friendly_ships = total
        else:
            total = info.enemy_ships + ships
            if total > 0:
                info.enemy_eta = (info.enemy_eta * info.enemy_ships + eta * ships) / total
            info.enemy_ships = total


def compute_fleet_transit(state: GameState) -> FleetTransitState:
    """Compute fleet transit info for all planets from current fleet positions."""
    transit = FleetTransitState()
    for f in state.fleets:
        best_planet = None
        best_eta = float("inf")
        for p in state.planets:
            eta = fleet_hits_planet(f, p)
            if eta is not None and eta < best_eta:
                best_eta = eta
                best_planet = p
        if best_planet is not None:
            is_friendly = f.owner == state.player
            transit.add_fleet(best_planet.id, float(f.ships), best_eta, is_friendly)
    return transit


# ── feature encoding ─────────────────────────────────────────────────────────

@dataclass
class SourceDecision:
    """All features needed for one source planet's transformer decision."""
    global_features: np.ndarray       # [GLOBAL_DIM]
    source_scalars: np.ndarray        # [SOURCE_SCALAR_DIM]
    source_position: np.ndarray       # [2]
    knn_scalars: np.ndarray           # [K, KNN_SCALAR_DIM]
    knn_positions: np.ndarray         # [K, 2]
    target_scalars: np.ndarray        # [T, TARGET_SCALAR_DIM]
    target_positions: np.ndarray      # [T, 2]
    target_mask: np.ndarray           # [T+2] bool — CLS + NoOp + targets
    # Metadata (not fed to network)
    target_planet_ids: list[int]
    target_angles: list[float]
    source_id: int
    remaining_ships: int


def build_global_features(state: GameState) -> np.ndarray:
    """Build [GLOBAL_DIM] global feature vector."""
    my_ships = 0.0
    my_prod = 0.0
    # Support up to 3 enemy players (indexed by relative order)
    enemy_ships = [0.0, 0.0, 0.0]
    enemy_prod = [0.0, 0.0, 0.0]
    enemy_ids: list[int] = []

    for p in state.planets:
        if p.owner == state.player:
            my_ships += p.ships
            my_prod += p.production
        elif p.owner >= 0:
            if p.owner not in enemy_ids:
                enemy_ids.append(p.owner)
            idx = enemy_ids.index(p.owner)
            if idx < 3:
                enemy_ships[idx] += p.ships
                enemy_prod[idx] += p.production

    # Add fleet ships
    for f in state.fleets:
        if f.owner == state.player:
            my_ships += f.ships
        elif f.owner >= 0:
            if f.owner not in enemy_ids:
                enemy_ids.append(f.owner)
            idx = enemy_ids.index(f.owner)
            if idx < 3:
                enemy_ships[idx] += f.ships

    max_s = max(my_ships, max(enemy_ships) if enemy_ships else 1.0, 1.0)
    max_p = max(my_prod, max(enemy_prod) if enemy_prod else 1.0, 1.0)

    return np.array([
        state.step / MAX_STEPS,
        my_ships / max_s,
        enemy_ships[0] / max_s,
        enemy_ships[1] / max_s,
        enemy_ships[2] / max_s,
        my_prod / max_p,
        enemy_prod[0] / max_p,
        enemy_prod[1] / max_p,
        enemy_prod[2] / max_p,
    ], dtype=np.float32)


def _select_targets(
    src: PlanetState,
    state: GameState,
    env_cfg: EnvConfig,
) -> list[PlanetState]:
    """Select up to max_targets planets as target candidates, sorted by distance."""
    others = [p for p in state.planets if p.id != src.id]
    others.sort(key=lambda p: math.hypot(p.x - src.x, p.y - src.y))
    return others[:env_cfg.max_targets]


def _select_knn(
    src: PlanetState,
    state: GameState,
    env_cfg: EnvConfig,
) -> list[PlanetState]:
    """Select K nearest neighbor planets (any ownership)."""
    others = [p for p in state.planets if p.id != src.id]
    others.sort(key=lambda p: math.hypot(p.x - src.x, p.y - src.y))
    return others[:env_cfg.k_neighbors]


def encode_source_decision(
    src: PlanetState,
    state: GameState,
    transit: FleetTransitState,
    env_cfg: EnvConfig,
) -> SourceDecision:
    """Encode all features for one source planet decision."""
    T = env_cfg.max_targets
    K = env_cfg.k_neighbors

    # Global features
    global_feat = build_global_features(state)

    # Source transit info
    src_transit = transit.get(src.id)

    # Source scalars
    source_scalars = np.array([
        src.radius / 5.0,
        src.production / env_cfg.max_production,
        math.log1p(src.ships) / 10.0,
        math.log1p(src_transit.enemy_ships) / 10.0,
        src_transit.enemy_eta / 50.0 if src_transit.enemy_ships > 0 else 0.0,
        math.log1p(src_transit.friendly_ships) / 10.0,
        src_transit.friendly_eta / 50.0 if src_transit.friendly_ships > 0 else 0.0,
    ], dtype=np.float32)

    source_position = np.array([src.x / BOARD_SIZE, src.y / BOARD_SIZE], dtype=np.float32)

    # KNN neighbors
    knn_planets = _select_knn(src, state, env_cfg)
    knn_scalars = np.zeros((K, KNN_SCALAR_DIM), dtype=np.float32)
    knn_positions = np.zeros((K, 2), dtype=np.float32)
    for i, kp in enumerate(knn_planets):
        knn_positions[i] = [kp.x / BOARD_SIZE, kp.y / BOARD_SIZE]
        knn_scalars[i] = [
            kp.radius / 5.0,
            kp.production / env_cfg.max_production,
            math.log1p(kp.ships) / 10.0,
            1.0 if kp.is_orbiting else 0.0,
        ]

    # Target planets
    targets = _select_targets(src, state, env_cfg)
    target_scalars = np.zeros((T, TARGET_SCALAR_DIM), dtype=np.float32)
    target_positions = np.zeros((T, 2), dtype=np.float32)
    # Mask: [CLS, NoOp, Target_0, ..., Target_{T-1}]
    target_mask = np.zeros(T + 2, dtype=bool)
    target_mask[0] = True   # CLS always valid
    target_mask[1] = True   # NoOp always valid

    target_planet_ids: list[int] = []
    target_angles: list[float] = []

    for i, tgt in enumerate(targets):
        dist = math.hypot(tgt.x - src.x, tgt.y - src.y)
        tgt_transit = transit.get(tgt.id)

        target_positions[i] = [tgt.x / BOARD_SIZE, tgt.y / BOARD_SIZE]
        target_scalars[i] = [
            1.0 if tgt.owner == -1 else 0.0,                               # neutral
            1.0 if tgt.owner == state.player else 0.0,                      # friendly
            1.0 if tgt.owner >= 0 and tgt.owner != state.player else 0.0,   # enemy
            dist / BOARD_SIZE,                                               # distance
            math.log1p(tgt.ships) / 10.0,                                   # ships
            tgt.production / env_cfg.max_production,                         # production
            math.log1p(tgt_transit.enemy_ships) / 10.0,                      # enemy transit
            tgt_transit.enemy_eta / 50.0 if tgt_transit.enemy_ships > 0 else 0.0,  # enemy eta
            math.log1p(tgt_transit.friendly_ships) / 10.0,                   # friendly transit
            tgt_transit.friendly_eta / 50.0 if tgt_transit.friendly_ships > 0 else 0.0,  # friendly eta
            1.0 if tgt.is_orbiting else 0.0,                                # orbiting
        ]

        # Compute angle (with sun avoidance)
        angle, _ = safe_angle(src.x, src.y, tgt.x, tgt.y)
        target_angles.append(angle)
        target_planet_ids.append(tgt.id)

        # Target is valid if we have ships and path doesn't cross sun
        crosses_sun = passes_through_sun(src.x, src.y, tgt.x, tgt.y)
        target_mask[i + 2] = not crosses_sun and src.ships > 0

    return SourceDecision(
        global_features=global_feat,
        source_scalars=source_scalars,
        source_position=source_position,
        knn_scalars=knn_scalars,
        knn_positions=knn_positions,
        target_scalars=target_scalars,
        target_positions=target_positions,
        target_mask=target_mask,
        target_planet_ids=target_planet_ids,
        target_angles=target_angles,
        source_id=src.id,
        remaining_ships=src.ships,
    )
