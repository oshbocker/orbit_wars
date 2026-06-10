"""Vendored public Kaggle Orbit Wars agents, for local benchmarking (the arena).

These are public competition notebooks (Kaggle notebooks default to Apache-2.0);
vendored 2026-06-10 for use as opponents/teachers/baselines. Sources and claimed
leaderboard scores (our team was at 736.7 when these were pulled):

    producer/               1287.1  slawekbiel via romantamrazov/orbit-wars-i-m-better
                                    (torch flow-diff planner: main.py + orbit_lite pkg)
    tamrazov_1224.py        1224    romantamrazov/orbit-star-wars-lb-max-1224
    distance_1100.py        1100    ykhnkf/distance-prioritized-agent-lb-max-score-1100
    shot_validator_hybrid.py ~1000? konbu17/orbit-wars-rule-base-ml-shot-validator-hybrid
                                    (v4 rule-base + reject-only MLP veto; weights.npz)
    enders_1000.py          1000+   zacharymaronek/orbit-wars-heuristic-agent-scored-1000
    ow_proto.py             ~1080   djenkivanov/orbit-wars-agent-ow-proto-passed-1-000
    reinforce_958.py        958.1   sigmaborov/lb-958-1-orbit-wars-2026-reinforce
                                    (pure rule-base despite the name)

Use load_agent(name) to get a FRESH agent callable per game: these agents keep
module-level state (fleet ledgers, movement caches), so reusing one callable across
games leaks state. In parallel runs, also use one process per game.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
import types
from pathlib import Path

_HERE = Path(__file__).resolve().parent

# name -> file (exec-style agents)
_FILE_AGENTS = {
    "tamrazov_1224": _HERE / "tamrazov_1224.py",
    "distance_1100": _HERE / "distance_1100.py",
    "shot_validator_hybrid": _HERE / "shot_validator_hybrid.py",
    "enders_1000": _HERE / "enders_1000.py",
    "ow_proto": _HERE / "ow_proto.py",
    "reinforce_958": _HERE / "reinforce_958.py",
}

EXTERNAL_AGENTS = ["producer", *_FILE_AGENTS]

_counter = itertools.count()


def _wrap(fn):
    """Normalize to agent(obs, config) regardless of the vendored signature."""
    if fn.__code__.co_argcount >= 2:
        return fn

    def agent(obs, config=None):
        return fn(obs)

    return agent


def load_agent(name: str):
    """Return a fresh agent(obs, config) callable for a vendored agent."""
    n = next(_counter)
    if name == "producer":
        pkg_dir = str(_HERE / "producer")
        if pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)
        modname = f"_ext_producer_{n}"
        spec = importlib.util.spec_from_file_location(modname, _HERE / "producer" / "main.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod  # required: dataclass introspection needs sys.modules
        spec.loader.exec_module(mod)
        return _wrap(mod.agent)
    path = _FILE_AGENTS[name]
    modname = f"_ext_{name}_{n}"
    mod = types.ModuleType(modname)
    mod.__file__ = str(path)
    sys.modules[modname] = mod  # required: frozen dataclasses look up sys.modules[__module__]
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)
    return _wrap(mod.agent)
