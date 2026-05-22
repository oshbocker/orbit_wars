"""Environment wrapper for V2 pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.game_types import GameState, parse_observation
from src.opponents import OpponentPolicy

from .config import V2Config
from .comet import comet_evacuation_moves
from .features import V2Features, encode_features
from .reward import compute_reward


@dataclass(slots=True)
class V2StepResult:
    features: V2Features
    reward: float
    done: bool
    info: dict[str, Any]


class V2OrbitWarsEnv:
    """Wraps kaggle_environments orbit_wars for V2 training."""

    def __init__(
        self,
        cfg: V2Config,
        opponent: OpponentPolicy,
        env_index: int = 0,
    ) -> None:
        self.cfg = cfg
        self.opponents: list[OpponentPolicy] = [opponent]
        self.env_index = env_index
        self.env: Any | None = None
        self.last_obs: Any = None
        self.last_opp_obs: list[Any] = []
        self.last_state: GameState | None = None
        self.prev_state: GameState | None = None
        self.episode_index = 0
        self.learner_player = 0
        self.num_players = 2

    def reset(
        self,
        seed: int | None = None,
        num_players: int | None = None,
        opponents: list[OpponentPolicy] | None = None,
    ) -> V2Features:
        """Reset environment, return V2Features for learner."""
        if opponents is not None:
            self.opponents = opponents
        if num_players is not None:
            self.num_players = num_players
        else:
            self.num_players = 2

        from kaggle_environments import make
        configuration: dict[str, Any] = {}
        if seed is not None:
            configuration["seed"] = int(seed)
            configuration["randomSeed"] = int(seed)

        # Side alternation
        if self.num_players == 2 and self.cfg.alternate_player_sides:
            self.learner_player = (self.env_index + self.episode_index) % 2
        elif self.num_players == 4:
            import random as _rng
            self.learner_player = _rng.randint(0, 3)
        else:
            self.learner_player = 0

        self.env = make("orbit_wars", configuration=configuration, debug=False)
        self.env.reset(num_agents=self.num_players)
        states = self.env.step([[] for _ in range(self.num_players)])

        self.last_obs = _extract_observation(states[self.learner_player])
        self.last_opp_obs = [
            _extract_observation(states[i])
            for i in range(self.num_players) if i != self.learner_player
        ]
        self.episode_index += 1

        state = parse_observation(self.last_obs)
        self.last_state = state
        self.prev_state = None

        return encode_features(state, self.cfg.env)

    def step(self, player_moves: list[list[float | int]]) -> V2StepResult:
        """Step environment with player's moves."""
        if self.env is None:
            raise RuntimeError("Call reset() before step().")

        # Comet evacuation
        state = self.last_state
        comet_ids = _get_comet_ids(self.last_obs)
        evac_moves, _ = comet_evacuation_moves(state, comet_ids, self.last_obs)

        # Combine evacuation + RL moves
        all_moves = evac_moves + player_moves

        # Build joint action
        joint_action: list[Any] = [[] for _ in range(self.num_players)]
        joint_action[self.learner_player] = all_moves

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

        # Parse new state
        self.prev_state = self.last_state
        new_state = parse_observation(self.last_obs)
        self.last_state = new_state

        # Compute reward
        terminal_reward = 0.0
        if done:
            terminal_reward = _terminal_reward_multi(states, self.learner_player)
        reward = compute_reward(
            self.prev_state, new_state, new_state.player,
            done, terminal_reward, self.cfg.reward,
        )

        features = encode_features(new_state, self.cfg.env)
        info = {
            "learner_player": self.learner_player,
            "num_players": self.num_players,
            "player_status": _extract_status(player_state),
        }
        return V2StepResult(features=features, reward=reward, done=done, info=info)


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
    pr = _extract_reward(states[learner_player])
    others = [_extract_reward(states[i]) for i in range(len(states)) if i != learner_player]
    if pr > 0.0 and any(o > 0.0 for o in others):
        return 0.0  # tie
    return pr


def _get_comet_ids(obs: Any) -> list[int] | None:
    if hasattr(obs, "comet_planet_ids"):
        ids = getattr(obs, "comet_planet_ids", None)
    elif isinstance(obs, dict):
        ids = obs.get("comet_planet_ids")
    else:
        return None
    if ids is None:
        return None
    return [int(x) for x in ids]
