"""Half-drainer FIXTURE bot (Isaiah @ Tufa, LB #1 1762 — replay diagnostic).

Sends ~half a garrison at a time (median send-fraction 0.52), one wave per source
planet per turn (~1.3-1.6 waves/turn). A deterministic, structurally NON-producer
measurement fixture for the off-mirror opponent-injection gate — NOT a competitor.
See ``agents/external/archetype_common.py`` for the shared planner + provenance.
"""

from __future__ import annotations

from agents.external.archetype_common import plan


def agent(obs, config=None) -> list:
    return plan(
        obs,
        targets_per_source=1,  # one cheap capture per source => ~1-2 waves/turn
        max_total_waves=999,
        drain_cap=0.6,  # keep reserves: capture-minimal sends => partial-send signature
        min_ships=6.0,
        min_fleet=5,
    )
