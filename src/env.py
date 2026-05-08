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
    """Wraps kaggle_environments orbit_wars for 2-player training."""

    def __init__(
        self,
        cfg: TrainConfig,
        opponent: OpponentPolicy,
        make_fn: Any | None = None,
        env_index: int = 0,
    ) -> None:
        self.cfg = cfg
        self.opponent = opponent
        self.make_fn = make_fn
        self.env_index = env_index
        self.env: Any | None = None
        self.last_obs: Any | None = None
        self.last_opp_obs: Any | None = None
        self.episode_index = 0
        self.learner_player = 0
        # Dense reward tracking
        self._prev_own_ships = 0.0
        self._prev_own_prod = 0.0

    def reset(self, seed: int | None = None) -> Any:
        """Reset environment, return raw observation for learner."""
        make_fn = self.make_fn or _default_make_fn()
        configuration: dict[str, Any] = {}
        if seed is not None:
            configuration["seed"] = int(seed)
            configuration["randomSeed"] = int(seed)

        if self.cfg.alternate_player_sides:
            self.learner_player = (self.env_index + self.episode_index) % 2
        else:
            self.learner_player = 0

        self.env = make_fn("orbit_wars", configuration=configuration, debug=False)
        self.env.reset(num_agents=2)
        states = self.env.step([[], []])

        self.last_obs = _extract_observation(states[self.learner_player])
        self.last_opp_obs = _extract_observation(states[1 - self.learner_player])
        self.episode_index += 1

        # Init dense reward baseline
        state = parse_observation(self.last_obs)
        self._prev_own_ships, self._prev_own_prod = _count_own(state)

        return self.last_obs

    def step(self, player_action: list[list[float | int]]) -> StepResult:
        if self.env is None:
            raise RuntimeError("Call reset() before step().")

        opponent_action = self.opponent.act(self.last_opp_obs)
        if self.learner_player == 0:
            joint_action = [player_action, opponent_action]
        else:
            joint_action = [opponent_action, player_action]

        states = self.env.step(joint_action)
        player_state = states[self.learner_player]
        opp_state = states[1 - self.learner_player]

        self.last_obs = _extract_observation(player_state)
        self.last_opp_obs = _extract_observation(opp_state)

        done = _extract_status(player_state) != "ACTIVE"

        # Compute reward
        reward = self._compute_reward(player_state, opp_state, done)

        info = {
            "learner_player": self.learner_player,
            "player_status": _extract_status(player_state),
            "opponent_status": _extract_status(opp_state),
        }
        return StepResult(obs=self.last_obs, reward=reward, done=done, info=info)

    def _compute_reward(self, player_state: Any, opp_state: Any, done: bool) -> float:
        reward_cfg = self.cfg.reward

        if done:
            return _terminal_reward(player_state, opp_state)

        if reward_cfg.sparse:
            return 0.0

        # Dense reward: delta in own ships and production
        state = parse_observation(self.last_obs)
        own_ships, own_prod = _count_own(state)
        delta_ships = own_ships - self._prev_own_ships
        delta_prod = own_prod - self._prev_own_prod
        self._prev_own_ships = own_ships
        self._prev_own_prod = own_prod

        return delta_ships * reward_cfg.dense_ship_coef + delta_prod * reward_cfg.dense_prod_coef


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


def _terminal_reward(player_state: Any, opp_state: Any) -> float:
    pr = _extract_reward(player_state)
    opr = _extract_reward(opp_state)
    if pr > 0.0 and opr > 0.0:
        return 0.0  # tie
    return pr


def _count_own(state: GameState) -> tuple[float, float]:
    ships = sum(p.ships for p in state.planets if p.owner == state.player)
    ships += sum(f.ships for f in state.fleets if f.owner == state.player)
    prod = sum(p.production for p in state.planets if p.owner == state.player)
    return float(ships), float(prod)
