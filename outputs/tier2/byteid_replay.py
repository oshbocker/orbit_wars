"""Byte-identity via FIXED obs-sequence replay (the harness is nondeterministic at
the full-game level, so replay identical inputs instead). Capture seat-0's obs
sequence from one reference game, then feed it through HEAD and NEW(ens=0) with
fresh memory. Identical inputs => identical action rows iff the code path is
equivalent. Also: replay NEW twice to confirm per-call determinism.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
ROOT = Path("/home/aeschbacher/git/orbit_wars")
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

torch.set_num_threads(1)
from agents import _normalize_obs, load_named_agent  # noqa: E402

_c = 0


def load(modfile, overrides):
    global _c
    _c += 1
    import dataclasses
    spec = importlib.util.spec_from_file_location(f"_m{_c}", ROOT / "agents" / "v5" / modfile)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"_m{_c}"] = mod
    spec.loader.exec_module(mod)
    base = mod.ProducerLiteConfig()
    cfg = dataclasses.replace(base, **overrides) if overrides else base
    mod._config_for = lambda pc: (mod.CONFIG_4P if int(pc) >= 4 else cfg)
    return mod


def capture_obs_seq(seed, opp):
    from kaggle_environments import make
    ref = load("main.py", {"contest_ensemble": 0})

    def a0(obs, config=None):
        return ref.agent(_normalize_obs(obs))
    env = make("orbit_wars", configuration={"randomSeed": seed})
    env.run([a0, load_named_agent(opp)])
    # seat-0 observation at each step
    return [st[0].observation for st in env.steps]


def replay(mod, obs_seq):
    mod._RUNTIME.reset()
    out = []
    for ob in obs_seq:
        out.append(repr(mod.agent(_normalize_obs(ob))))
    return out


def main():
    seeds = [1000, 1001, 1002, 1003]
    total_mism = 0
    for s in seeds:
        obs_seq = capture_obs_seq(s, "producer")
        head = load("_main_head.py", {})            # HEAD default (v5.4)
        new0 = load("main.py", {"contest_ensemble": 0})
        r_head = replay(head, obs_seq)
        r_new = replay(new0, obs_seq)
        r_new2 = replay(new0, obs_seq)              # determinism of NEW
        det = r_new == r_new2
        same = r_head == r_new
        if not same:
            total_mism += 1
            for i, (a, b) in enumerate(zip(r_head, r_new)):
                if a != b:
                    print(f"  seed {s} DIFF turn {i}:\n   HEAD {a}\n   NEW  {b}")
                    break
        print(f"seed {s}: HEAD==NEW {'OK' if same else 'MISMATCH'} | NEW det {'OK' if det else 'NONDET'} ({len(obs_seq)} turns)")
    print(f"\nreplay byte-identity mismatches: {total_mism}/{len(seeds)}")


if __name__ == "__main__":
    main()
