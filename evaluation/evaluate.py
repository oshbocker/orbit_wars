"""
Evaluation utilities for Orbit Wars agents.

All functions accept any Kaggle-compatible agent callable (obs, config) -> moves.
"""

from __future__ import annotations

from collections.abc import Callable

from kaggle_environments import make


def run_games(
    agent_a: Callable,
    agent_b: Callable | str,
    n_games: int = 20,
    verbose: bool = False,
) -> dict:
    """
    Run n_games between agent_a (player 0) and agent_b (player 1).

    Returns
    -------
    dict with keys: wins, losses, ties, n_games, win_rate, loss_rate, tie_rate
    """
    wins = losses = ties = 0
    for g in range(n_games):
        env = make("orbit_wars", debug=False)
        env.run([agent_a, agent_b])
        reward = env.steps[-1][0].reward
        if reward is None or reward == 0:
            ties += 1
            result = "T"
        elif reward > 0:
            wins += 1
            result = "W"
        else:
            losses += 1
            result = "L"
        if verbose:
            steps = len(env.steps)
            print(f"  game {g + 1:3d}/{n_games}  {result}  ({steps} steps)")

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n_games": n_games,
        "win_rate": wins / n_games,
        "loss_rate": losses / n_games,
        "tie_rate": ties / n_games,
    }


def print_results(label_a: str, label_b: str, results: dict) -> None:
    w = results["win_rate"]
    l = results["loss_rate"]
    t = results["tie_rate"]
    n = results["n_games"]
    print(f"{label_a:20s} vs {label_b:20s}  |  W:{w:.0%}  L:{l:.0%}  T:{t:.0%}  (n={n})")


def head_to_head(
    agents: dict[str, Callable],
    n_games: int = 20,
    verbose: bool = False,
) -> dict:
    """
    Run every pair combination and return a results matrix.

    Parameters
    ----------
    agents : dict mapping name → agent callable
    n_games : games per matchup

    Returns
    -------
    dict[str, dict[str, dict]] — results[(a, b)] = run_games output
    """
    names = list(agents.keys())
    results = {}
    for i, a_name in enumerate(names):
        for b_name in names[i + 1 :]:
            if verbose:
                print(f"\n{a_name} vs {b_name}:")
            r = run_games(agents[a_name], agents[b_name], n_games=n_games, verbose=verbose)
            results[(a_name, b_name)] = r
            print_results(a_name, b_name, r)
    return results


def benchmark(agent: Callable, agent_name: str = "agent", n_games: int = 20) -> None:
    """Convenience: run agent vs random and producer, print a summary table."""
    from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent

    from agents import load_named_agent

    print(f"\n{'=' * 60}")
    print(f"  Benchmark: {agent_name}")
    print(f"{'=' * 60}")
    for opp_name, opp in [("random", random_agent), ("producer", load_named_agent("producer"))]:
        r = run_games(agent, opp, n_games=n_games)
        print_results(agent_name, opp_name, r)
    print()
