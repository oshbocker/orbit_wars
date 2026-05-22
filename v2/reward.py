"""Reward computation for V2 pipeline."""
from __future__ import annotations

from src.game_types import GameState

from .config import V2RewardConfig


def compute_reward(
    prev_state: GameState | None,
    curr_state: GameState,
    player: int,
    done: bool,
    terminal_reward: float,
    cfg: V2RewardConfig,
) -> float:
    """Compute reward for a single step.

    Args:
        prev_state: Previous game state (None on first step).
        curr_state: Current game state.
        player: Player index.
        done: Whether episode ended.
        terminal_reward: Kaggle terminal reward (+1/-1/0).
        cfg: Reward config.
    """
    if done:
        return terminal_reward

    mode = cfg.reward_mode
    if mode == "sparse":
        return 0.0

    if prev_state is None:
        return 0.0

    # Early-game production bonus multiplier
    prod_mult = 1.0
    if cfg.early_prod_bonus > 0 and cfg.early_prod_bonus_steps > 0:
        t = max(0.0, 1.0 - curr_state.step / cfg.early_prod_bonus_steps)
        prod_mult = 1.0 + cfg.early_prod_bonus * t

    if mode == "dense_absolute":
        prev_ships, prev_prod = _count_own(prev_state, player)
        curr_ships, curr_prod = _count_own(curr_state, player)
        return ((curr_ships - prev_ships) * cfg.dense_ship_coef
                + (curr_prod - prev_prod) * cfg.dense_prod_coef * prod_mult)

    # dense_relative
    prev_all_ships = _count_all_ships(prev_state)
    curr_all_ships = _count_all_ships(curr_state)
    prev_own = prev_all_ships.get(player, 0.0)
    curr_own = curr_all_ships.get(player, 0.0)
    prev_best_enemy = max((s for p, s in prev_all_ships.items() if p != player), default=0.0)
    curr_best_enemy = max((s for p, s in curr_all_ships.items() if p != player), default=0.0)
    delta_ship_gap = (curr_own - curr_best_enemy) - (prev_own - prev_best_enemy)

    prev_all_prod = _count_all_production(prev_state)
    curr_all_prod = _count_all_production(curr_state)
    prev_own_prod = prev_all_prod.get(player, 0.0)
    curr_own_prod = curr_all_prod.get(player, 0.0)
    prev_best_enemy_prod = max((s for p, s in prev_all_prod.items() if p != player), default=0.0)
    curr_best_enemy_prod = max((s for p, s in curr_all_prod.items() if p != player), default=0.0)
    delta_prod_gap = (curr_own_prod - curr_best_enemy_prod) - (prev_own_prod - prev_best_enemy_prod)

    return delta_ship_gap * cfg.dense_ship_coef + delta_prod_gap * cfg.dense_prod_coef * prod_mult


def _count_own(state: GameState, player: int) -> tuple[float, float]:
    ships = sum(p.ships for p in state.planets if p.owner == player)
    ships += sum(f.ships for f in state.fleets if f.owner == player)
    prod = sum(p.production for p in state.planets if p.owner == player)
    return float(ships), float(prod)


def _count_all_ships(state: GameState) -> dict[int, float]:
    counts: dict[int, float] = {}
    for p in state.planets:
        if p.owner >= 0:
            counts[p.owner] = counts.get(p.owner, 0.0) + p.ships
    for f in state.fleets:
        if f.owner >= 0:
            counts[f.owner] = counts.get(f.owner, 0.0) + f.ships
    return counts


def _count_all_production(state: GameState) -> dict[int, float]:
    counts: dict[int, float] = {}
    for p in state.planets:
        if p.owner >= 0:
            counts[p.owner] = counts.get(p.owner, 0.0) + p.production
    return counts
