from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np
import torch

from .config import TrainConfig
from .features import (
    FleetTransitState,
    SourceDecision,
    compute_fleet_transit,
    encode_source_decision,
)
from .game_types import GameState, parse_observation
from .policy import TransformerPolicy
from .ppo import sample_actions


class OpponentPolicy(Protocol):
    def act(self, observation: Any) -> list[list[float | int]]:
        ...


class CompetitiveOpponent:
    def __init__(self) -> None:
        from agents.competitive import agent as _competitive_agent

        self._agent = _competitive_agent

    def act(self, observation: Any) -> list[list[float | int]]:
        result = self._agent(observation)
        return list(result) if result else []


class KaggleRandomOpponent:
    def __init__(self) -> None:
        from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent

        self._agent = random_agent

    def act(self, observation: Any) -> list[list[float | int]]:
        payload = {
            "player": _obs_get(observation, "player", 0),
            "planets": list(_obs_get(observation, "planets", [])),
        }
        return list(self._agent(payload))


class SelfPlayOpponent:
    def __init__(self, cfg: TrainConfig, device: torch.device, deterministic: bool = True) -> None:
        self.cfg = cfg
        self.device = device
        self.deterministic = deterministic
        self.policy = TransformerPolicy(cfg.model, cfg.env).to(device)
        self.policy.eval()

    def sync_from(self, source_policy: TransformerPolicy) -> None:
        self.policy.load_state_dict(source_policy.state_dict())
        self.policy.eval()

    def act(self, observation: Any) -> list[list[float | int]]:
        state = observation if isinstance(observation, GameState) else parse_observation(observation)
        my_planets = sorted(
            [p for p in state.planets if p.owner == state.player],
            key=lambda p: -p.ships,
        )
        if not my_planets:
            return []

        transit = compute_fleet_transit(state)
        moves: list[list[float | int]] = []

        for src in my_planets:
            decision = encode_source_decision(src, state, transit, self.cfg.env)
            with torch.inference_mode():
                outputs = self.policy(
                    torch.from_numpy(decision.global_features).unsqueeze(0).to(self.device),
                    torch.from_numpy(decision.source_scalars).unsqueeze(0).to(self.device),
                    torch.from_numpy(decision.source_position).unsqueeze(0).to(self.device),
                    torch.from_numpy(decision.knn_scalars).unsqueeze(0).to(self.device),
                    torch.from_numpy(decision.knn_positions).unsqueeze(0).to(self.device),
                    torch.from_numpy(decision.target_scalars).unsqueeze(0).to(self.device),
                    torch.from_numpy(decision.target_positions).unsqueeze(0).to(self.device),
                    torch.from_numpy(decision.target_mask).unsqueeze(0).to(self.device),
                )
                sampled = sample_actions(outputs, deterministic=self.deterministic)

            tgt_idx = int(sampled.target_index.item())
            if tgt_idx == 0:
                continue
            target_offset = tgt_idx - 1
            if target_offset >= len(decision.target_planet_ids):
                continue

            frac_bin = int(sampled.fraction_bin.item())
            fraction = self.cfg.env.ship_fractions[frac_bin]
            ships = int(src.ships * fraction)
            if ships <= 0:
                continue

            target_id = decision.target_planet_ids[target_offset]
            angle = decision.target_angles[target_offset]
            moves.append([src.id, float(angle), ships])

            # Update transit and deduct ships
            speed = 1.0 + 5.0 * (math.log(max(ships, 2)) / math.log(1000)) ** 1.5
            tgt_planet = state.planets_by_id.get(target_id)
            if tgt_planet:
                dist = math.hypot(src.x - tgt_planet.x, src.y - tgt_planet.y)
                eta = dist / max(speed, 0.1)
                transit.add_fleet(target_id, float(ships), eta, is_friendly=True)
            src.ships -= ships

        return moves


def build_opponent(
    name: str,
    cfg: TrainConfig | None = None,
    device: torch.device | None = None,
) -> OpponentPolicy:
    if name == "competitive":
        return CompetitiveOpponent()
    if name == "random":
        return KaggleRandomOpponent()
    if name == "self":
        if cfg is None or device is None:
            raise ValueError("cfg and device required for self-play opponent")
        return SelfPlayOpponent(cfg, device=device, deterministic=cfg.self_play_deterministic)
    raise ValueError(f"Unknown opponent: {name}")


def _obs_get(observation: Any, key: str, default: Any) -> Any:
    if isinstance(observation, dict):
        return observation.get(key, default)
    return getattr(observation, key, default)
