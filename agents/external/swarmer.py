"""Swarmer FIXTURE bot (213tubo, LB 1536 — replay diagnostic).

Many small waves per turn (~14.2 waves/turn observed) at a moderate per-source
drain-fraction (median 0.67), spread across several nearby targets — swarm micro-
logistics, structurally distinct from producer's ~1.5 full-drain waves/turn. A
deterministic measurement fixture for the off-mirror opponent-injection gate, NOT a
competitor. See ``agents/external/archetype_common.py`` for the shared planner.
"""

from __future__ import annotations

from agents.external.archetype_common import plan


def agent(obs, config=None) -> list:
    return plan(
        obs,
        targets_per_source=4,  # attack several targets at once
        max_total_waves=14,  # ~14.2 waves/turn from the replay diagnostic
        drain_cap=0.85,  # commit more, but still leave a small reserve
        min_ships=6.0,
        min_fleet=3,  # small fleets => the swarm signature
        split_k=4,  # each capture = 4 aggregating sub-fleets => many small waves
    )
