from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import RewardConfig, TrainConfig
from .game_types import GameState, parse_observation
from .opponents import OpponentPolicy


@dataclass(slots=True)
class StepResult:
    obs: Any
    reward: float
    done: bool
    info: dict[str, Any]


class OrbitWarsEnv:
    """Wraps kaggle_environments orbit_wars for multi-player training."""

    def __init__(
        self,
        cfg: TrainConfig,
        opponent: OpponentPolicy,
        make_fn: Any | None = None,
        env_index: int = 0,
    ) -> None:
        self.cfg = cfg
        self.opponents: list[OpponentPolicy] = [opponent]
        self.make_fn = make_fn
        self.env_index = env_index
        self.env: Any | None = None
        self.last_obs: Any | None = None
        self.last_opp_obs: list[Any] = []
        self.episode_index = 0
        self.learner_player = 0
        self.num_players = 2
        # Dense reward tracking
        self._prev_own_ships = 0.0
        self._prev_own_prod = 0.0
        self._prev_ship_gap = 0.0

    def reset(
        self,
        seed: int | None = None,
        num_players: int | None = None,
        opponents: list[OpponentPolicy] | None = None,
    ) -> Any:
        """Reset environment, return raw observation for learner."""
        if opponents is not None:
            self.opponents = opponents
        if num_players is not None:
            self.num_players = num_players
        else:
            self.num_players = 2

        make_fn = self.make_fn or _default_make_fn()
        configuration: dict[str, Any] = {}
        if seed is not None:
            configuration["seed"] = int(seed)
            configuration["randomSeed"] = int(seed)

        if self.num_players == 2 and self.cfg.alternate_player_sides:
            self.learner_player = (self.env_index + self.episode_index) % 2
        elif self.num_players == 4:
            import random as _rng
            self.learner_player = _rng.randint(0, 3)
        else:
            self.learner_player = 0

        self.env = make_fn("orbit_wars", configuration=configuration, debug=False)
        self.env.reset(num_agents=self.num_players)
        states = self.env.step([[] for _ in range(self.num_players)])

        self.last_obs = _extract_observation(states[self.learner_player])
        self.last_opp_obs = [
            _extract_observation(states[i])
            for i in range(self.num_players) if i != self.learner_player
        ]
        self.episode_index += 1

        # Init dense reward baseline
        state = parse_observation(self.last_obs)
        self._prev_own_ships, self._prev_own_prod = _count_own(state)
        all_ships = _count_all_ships(state)
        own = all_ships.get(state.player, 0.0)
        best_enemy = max((s for p, s in all_ships.items() if p != state.player), default=0.0)
        self._prev_ship_gap = own - best_enemy

        return self.last_obs

    def step(self, player_action: list[list[float | int]]) -> StepResult:
        if self.env is None:
            raise RuntimeError("Call reset() before step().")

        # Build joint action for all players
        joint_action: list[Any] = [[] for _ in range(self.num_players)]
        joint_action[self.learner_player] = player_action

        opp_idx = 0
        for i in range(self.num_players):
            if i == self.learner_player:
                continue
            opp = self.opponents[opp_idx % len(self.opponents)]
            joint_action[i] = opp.act(self.last_opp_obs[opp_idx])
            opp_idx += 1

        states = self.env.step(joint_action)
        player_state = states[self.learner_player]

        self.last_obs = _extract_observation(player_state)
        self.last_opp_obs = [
            _extract_observation(states[i])
            for i in range(self.num_players) if i != self.learner_player
        ]

        done = _extract_status(player_state) != "ACTIVE"

        # Compute reward
        reward = self._compute_reward(states, done)

        info = {
            "learner_player": self.learner_player,
            "num_players": self.num_players,
            "player_status": _extract_status(player_state),
        }
        return StepResult(obs=self.last_obs, reward=reward, done=done, info=info)

    def _compute_reward(self, states: list[Any], done: bool) -> float:
        reward_cfg = self.cfg.reward
        mode = reward_cfg.reward_mode

        if done:
            return _terminal_reward_multi(states, self.learner_player)

        if mode == "sparse":
            return 0.0

        state = parse_observation(self.last_obs)

        if mode == "dense_absolute":
            own_ships, own_prod = _count_own(state)
            delta_ships = own_ships - self._prev_own_ships
            delta_prod = own_prod - self._prev_own_prod
            self._prev_own_ships = own_ships
            self._prev_own_prod = own_prod
            return delta_ships * reward_cfg.dense_ship_coef + delta_prod * reward_cfg.dense_prod_coef

        # dense_relative: delta(our_ships - best_enemy_ships) * coef
        all_ships = _count_all_ships(state)
        own = all_ships.get(state.player, 0.0)
        best_enemy = max((s for p, s in all_ships.items() if p != state.player), default=0.0)
        ship_gap = own - best_enemy
        delta_gap = ship_gap - self._prev_ship_gap
        self._prev_ship_gap = ship_gap
        # Also update absolute tracking for consistency
        self._prev_own_ships = own
        return delta_gap * reward_cfg.dense_ship_coef


def _default_make_fn() -> Any:
    from kaggle_environments import make
    return make


def _extract_observation(state: Any) -> Any:
    if isinstance(state, dict):
        return state.get("observation")
    return getattr(state, "observation")


def _extract_status(state: Any) -> str:
    if isinstance(state, dict):
        return str(state.get("status", "UNKNOWN"))
    return str(getattr(state, "status", "UNKNOWN"))


def _extract_reward(state: Any) -> float:
    if isinstance(state, dict):
        value = state.get("reward", 0.0)
    else:
        value = getattr(state, "reward", 0.0)
    return 0.0 if value is None else float(value)


def _terminal_reward_multi(states: list[Any], learner_player: int) -> float:
    """Terminal reward using Kaggle rewards — works for 2p and 4p."""
    pr = _extract_reward(states[learner_player])
    # Check if any other player also won (tie)
    others = [_extract_reward(states[i]) for i in range(len(states)) if i != learner_player]
    if pr > 0.0 and any(o > 0.0 for o in others):
        return 0.0  # tie
    return pr


def _count_own(state: GameState) -> tuple[float, float]:
    ships = sum(p.ships for p in state.planets if p.owner == state.player)
    ships += sum(f.ships for f in state.fleets if f.owner == state.player)
    prod = sum(p.production for p in state.planets if p.owner == state.player)
    return float(ships), float(prod)


def _count_all_ships(state: GameState) -> dict[int, float]:
    """Total ships per player (planets + fleets)."""
    counts: dict[int, float] = {}
    for p in state.planets:
        if p.owner >= 0:
            counts[p.owner] = counts.get(p.owner, 0.0) + p.ships
    for f in state.fleets:
        if f.owner >= 0:
            counts[f.owner] = counts.get(f.owner, 0.0) + f.ships
    return counts
