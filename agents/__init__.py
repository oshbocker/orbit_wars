"""Named agent loading for benchmarks, opponents, and teachers.

The apex/hybrid rule-based agents were retired 2026-06-11 (design + post-mortem:
`rl_research/EXPLORED_AND_ABANDONED.md` Cluster 5). The strong rule-based pool is
now `agents/external/` (vendored public agents, producer = the 1287-tier base) and
`agents/v5/` (our producer fork — the shipped competition agent).
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_counter = itertools.count()

_PLANET_KEYS = ("id", "owner", "x", "y", "radius", "ships", "production")
_FLEET_KEYS = ("id", "owner", "x", "y", "angle", "from_planet_id", "ships")


def _normalize_obs(obs):
    """Convert fast_env dict-style planet/fleet rows to Kaggle list rows.

    The vendored public agents index rows positionally (``p[:7]``), which works on
    the real Kaggle env but not on v2.fast_env observations (dict rows). Real-env
    observations pass through untouched.
    """
    if not isinstance(obs, dict):
        return obs
    planets = obs.get("planets") or []
    if not (planets and isinstance(planets[0], dict)):
        return obs
    out = dict(obs)
    out["planets"] = [[p[k] for k in _PLANET_KEYS] for p in planets]
    out["fleets"] = [[f[k] for k in _FLEET_KEYS] for f in (obs.get("fleets") or [])]
    ip = obs.get("initial_planets") or []
    if ip and isinstance(ip[0], dict):
        out["initial_planets"] = [[p[k] for k in _PLANET_KEYS] for p in ip]
    # Struct, not plain dict: some vendored agents read obs.angular_velocity /
    # obs.step attribute-style (real-env Struct semantics).
    from kaggle_environments.utils import Struct

    return Struct(**out)


def load_named_agent(name: str):
    """Return a FRESH agent(obs, config) callable for a named agent.

    Names: "v5" (our producer fork), "producer" / "tamrazov_1224" / ... (vendored
    public agents from agents.external), or "random" (Kaggle built-in). Fresh-load
    per game — these agents keep module-level state (movement caches, ledgers).
    """
    if name == "v5":
        main_py = _HERE / "v5" / "main.py"
        modname = f"_v5_{next(_counter)}"
        spec = importlib.util.spec_from_file_location(modname, main_py)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
        fn = mod.agent

        def v5_agent(obs, config=None):
            return fn(_normalize_obs(obs))

        return v5_agent
    if name == "random":
        from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent

        def random_wrapped(obs, config=None):
            return random_agent(_normalize_obs(obs))

        return random_wrapped
    from agents.external import load_agent

    inner = load_agent(name)

    def named_agent(obs, config=None):
        return inner(_normalize_obs(obs), config)

    return named_agent


__all__ = ["load_named_agent"]
