"""
Gymnasium environment wrapper for Orbit Wars (single-agent, 2-player).

The RL agent plays as player 0 against a fixed opponent callable.

Observation space  — flat Box(683,) float32:
  [0:3]          global: step/500, angular_velocity/0.05, n_comets/4
  [3:283]        MAX_PLANETS × 7: is_mine, is_enemy, is_neutral,
                                   x/100, y/100, log1p(ships)/10, production/5
  [283:683]      MAX_FLEETS × 5:  is_mine, is_enemy,
                                   x/100, y/100, log1p(ships)/10

Action space — MultiDiscrete([MAX_OWN, MAX_PLANETS, N_FRACTIONS]):
  own_slot    : index into my planets sorted by id  (no-op if slot ≥ len(my_planets))
  target_slot : index into all planets              (no-op if same as src or out of range)
  frac_bin    : 0→25%, 1→50%, 2→75%, 3→100% of available ships

One action = at most one fleet launch per step.

Reward shaping (dense):
  +Δships      × 0.001   (net change in total own ships)
  +Δproduction × 0.005   (net change in owned production)
  +kaggle_reward (±1)    at terminal step
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from kaggle_environments import make as kg_make

# ── constants ─────────────────────────────────────────────────────────────
MAX_PLANETS: int = 40
MAX_FLEETS: int = 80
MAX_OWN: int = 12
N_FRACTIONS: int = 4
SHIP_FRACTIONS: list[float] = [0.25, 0.50, 0.75, 1.00]

OBS_GLOBAL = 3
OBS_PER_PLANET = 7
OBS_PER_FLEET = 5
OBS_DIM: int = OBS_GLOBAL + MAX_PLANETS * OBS_PER_PLANET + MAX_FLEETS * OBS_PER_FLEET  # 683


def _random_opponent(obs, config=None) -> list:
    """Built-in random opponent so callers can pass opponent='random'."""
    import random
    player = obs.player if hasattr(obs, "player") else obs.get("player", 0)
    raw_p = obs.planets if hasattr(obs, "planets") else obs.get("planets", [])
    mine = [p for p in raw_p if p[1] == player and p[5] > 2]
    if not mine:
        return []
    src = random.choice(mine)
    targets = [p for p in raw_p if p[1] != player]
    if not targets:
        return []
    tgt = random.choice(targets)
    angle = math.atan2(tgt[3] - src[3], tgt[2] - src[2])
    return [[src[0], angle, max(1, random.randint(1, src[5]))]]


class OrbitWarsEnv(gym.Env):
    """
    Single-agent Gymnasium wrapper.  The RL agent plays as player 0.

    Parameters
    ----------
    opponent : callable or "random" or "competitive"
        Agent function called each step to produce opponent moves.
    reward_shaping : bool
        Whether to add dense intermediate rewards on top of terminal ±1.
    seed : int | None
        Fixed map seed for reproducibility (None = random each reset).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        opponent: Callable | str = "random",
        reward_shaping: bool = True,
        seed: int | None = None,
    ):
        super().__init__()

        # Resolve string opponents
        if opponent == "random":
            self._opponent_fn = _random_opponent
        elif opponent == "competitive":
            from agents.competitive import agent as competitive_agent
            self._opponent_fn = competitive_agent
        elif callable(opponent):
            self._opponent_fn = opponent
        else:
            raise ValueError(f"opponent must be callable, 'random', or 'competitive'; got {opponent!r}")

        self.reward_shaping = reward_shaping
        self._seed = seed

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete(
            [MAX_OWN, MAX_PLANETS, N_FRACTIONS]
        )

        self._env = None
        self._n_steps = 0          # steps taken so far this episode
        self._done = False
        self._prev_ships = 0.0
        self._prev_prod = 0.0

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cfg = {"seed": seed if seed is not None else self._seed}
        if cfg["seed"] is None:
            cfg = {}

        self._env = kg_make("orbit_wars", configuration=cfg, debug=False)
        self._env.reset()
        self._n_steps = 0
        self._done = False

        obs_raw = self._env.steps[0][0].observation
        self._prev_ships, self._prev_prod = self._ship_prod_totals(obs_raw, player=0)
        return self._encode(obs_raw, player=0), {}

    def step(self, action: np.ndarray):
        assert not self._done, "Episode is done; call reset()."

        obs_raw = self._env.steps[self._n_steps][0].observation
        opp_raw = self._env.steps[self._n_steps][1].observation

        my_move = self._decode_action(obs_raw, int(action[0]), int(action[1]), int(action[2]))
        opp_move = self._opponent_fn(opp_raw)

        try:
            self._env.step([my_move, opp_move])
        except Exception as exc:
            # Malformed action or engine error — treat as terminal
            self._done = True
            return self._encode(obs_raw, player=0), 0.0, True, False, {"error": str(exc)}

        self._n_steps += 1
        next_raw = self._env.steps[self._n_steps][0].observation
        status = self._env.steps[self._n_steps][0].status
        terminated = status in ("DONE", "INVALID", "TIMEOUT", "ERROR")

        reward = 0.0
        if self.reward_shaping:
            reward += self._shape_reward(next_raw, player=0)
        if terminated:
            kaggle_r = float(self._env.steps[self._n_steps][0].reward or 0.0)
            reward += kaggle_r
            self._done = True

        return self._encode(next_raw, player=0), reward, terminated, False, {}

    def render(self):
        if self._env is not None:
            self._env.render(mode="ipython", width=800, height=600)

    # ── public helpers ─────────────────────────────────────────────────────

    def encode(self, obs_raw, player: int = 0) -> np.ndarray:
        """Public alias for use by RLAgent when generating submissions."""
        return self._encode(obs_raw, player)

    def decode_action(self, obs_raw, own_slot: int, tgt_slot: int, frac_bin: int) -> list:
        """Public alias used by RLAgent."""
        return self._decode_action(obs_raw, own_slot, tgt_slot, frac_bin)

    # ── private helpers ────────────────────────────────────────────────────

    def _decode_action(self, obs_raw, own_slot: int, tgt_slot: int, frac_bin: int) -> list:
        """Map (own_slot, tgt_slot, frac_bin) → [[planet_id, angle, ships]] or []."""
        player = _player(obs_raw)
        planets = [_make_planet(p) for p in _raw_planets(obs_raw)]
        my_planets = sorted([p for p in planets if p.owner == player], key=lambda p: p.id)

        if own_slot >= len(my_planets) or tgt_slot >= len(planets):
            return []

        src = my_planets[own_slot]
        tgt = planets[tgt_slot]
        if tgt.id == src.id:
            return []

        ships = max(1, int(src.ships * SHIP_FRACTIONS[frac_bin]))
        if ships >= src.ships:
            ships = max(1, src.ships - 1)

        angle = math.atan2(tgt.y - src.y, tgt.x - src.x)
        return [[src.id, angle, ships]]

    def _encode(self, obs_raw, player: int) -> np.ndarray:
        vec = np.zeros(OBS_DIM, dtype=np.float32)

        raw_planets = _raw_planets(obs_raw)
        raw_fleets = _raw_fleets(obs_raw)
        step = _step(obs_raw)
        av = _angular_velocity(obs_raw)
        comet_ids = set(_comet_ids(obs_raw))

        vec[0] = step / 500.0
        vec[1] = av / 0.05
        vec[2] = len(comet_ids) / 4.0

        base = OBS_GLOBAL
        for i, p_raw in enumerate(raw_planets[:MAX_PLANETS]):
            p = _make_planet(p_raw)
            b = base + i * OBS_PER_PLANET
            vec[b + 0] = 1.0 if p.owner == player else 0.0
            vec[b + 1] = 1.0 if (p.owner >= 0 and p.owner != player) else 0.0
            vec[b + 2] = 1.0 if p.owner == -1 else 0.0
            vec[b + 3] = p.x / 100.0
            vec[b + 4] = p.y / 100.0
            vec[b + 5] = math.log1p(p.ships) / 10.0
            vec[b + 6] = p.production / 5.0

        base = OBS_GLOBAL + MAX_PLANETS * OBS_PER_PLANET
        for i, f_raw in enumerate(raw_fleets[:MAX_FLEETS]):
            f = _make_fleet(f_raw)
            b = base + i * OBS_PER_FLEET
            vec[b + 0] = 1.0 if f.owner == player else 0.0
            vec[b + 1] = 1.0 if f.owner != player else 0.0
            vec[b + 2] = f.x / 100.0
            vec[b + 3] = f.y / 100.0
            vec[b + 4] = math.log1p(f.ships) / 10.0

        return vec

    def _shape_reward(self, obs_raw, player: int) -> float:
        ships, prod = self._ship_prod_totals(obs_raw, player)
        reward = (ships - self._prev_ships) * 0.001 + (prod - self._prev_prod) * 0.005
        self._prev_ships, self._prev_prod = ships, prod
        return reward

    def _ship_prod_totals(self, obs_raw, player: int) -> tuple[float, float]:
        planets = [_make_planet(p) for p in _raw_planets(obs_raw)]
        fleets = [_make_fleet(f) for f in _raw_fleets(obs_raw)]
        planet_ships = sum(p.ships for p in planets if p.owner == player)
        fleet_ships = sum(f.ships for f in fleets if f.owner == player)
        prod = sum(p.production for p in planets if p.owner == player)
        return float(planet_ships + fleet_ships), float(prod)


# ── observation accessors (handle both dict and namespace) ─────────────────

def _player(obs) -> int:
    return obs.player if hasattr(obs, "player") else obs.get("player", 0)

def _raw_planets(obs) -> list:
    return obs.planets if hasattr(obs, "planets") else obs.get("planets", [])

def _raw_fleets(obs) -> list:
    return obs.fleets if hasattr(obs, "fleets") else obs.get("fleets", [])

def _step(obs) -> int:
    return obs.step if hasattr(obs, "step") else obs.get("step", 0)

def _angular_velocity(obs) -> float:
    return obs.angular_velocity if hasattr(obs, "angular_velocity") else obs.get("angular_velocity", 0.0)

def _comet_ids(obs) -> list:
    return obs.comet_planet_ids if hasattr(obs, "comet_planet_ids") else obs.get("comet_planet_ids", [])


# ── lightweight data classes ───────────────────────────────────────────────

class _P:
    __slots__ = ("id", "owner", "x", "y", "radius", "ships", "production")
    def __init__(self, id, owner, x, y, radius, ships, production):
        self.id = id; self.owner = owner; self.x = x; self.y = y
        self.radius = radius; self.ships = ships; self.production = production

class _F:
    __slots__ = ("id", "owner", "x", "y", "angle", "from_planet_id", "ships")
    def __init__(self, id, owner, x, y, angle, from_planet_id, ships):
        self.id = id; self.owner = owner; self.x = x; self.y = y
        self.angle = angle; self.from_planet_id = from_planet_id; self.ships = ships


def _make_planet(p):
    """Parse a planet from Struct (attribute), dict, or list/tuple."""
    if hasattr(p, "production"):
        return _P(p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production)
    if isinstance(p, dict):
        return _P(p["id"], p["owner"], p["x"], p["y"], p["radius"], p["ships"], p["production"])
    return _P(*p)


def _make_fleet(f):
    """Parse a fleet from Struct (attribute), dict, or list/tuple."""
    if hasattr(f, "from_planet_id"):
        return _F(f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships)
    if isinstance(f, dict):
        return _F(f["id"], f["owner"], f["x"], f["y"], f["angle"], f["from_planet_id"], f["ships"])
    return _F(*f)
