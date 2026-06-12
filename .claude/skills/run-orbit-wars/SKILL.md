---
name: run-orbit-wars
description: Run, smoke-test, screenshot, evaluate, or train the Orbit Wars Kaggle agents — play a real game between named agents (v5, producer, ...), render/screenshot an HTML replay, run the arena gate, eval a checkpoint, run the test suites, do a 1-iteration ExIt training smoke, or build the submission bundle.
---

# Run Orbit Wars

Kaggle simulation-competition repo: rule-based + RL agents that play Orbit Wars
on the real `kaggle_environments` engine. There is no server or GUI app — the
"app" is *playing games*, and the driver is
`.claude/skills/run-orbit-wars/driver.py`, which runs one real game between any
two/four named agents and optionally dumps the animated HTML replay.

All paths are relative to the repo root. **Run everything from the repo root
with `uv run`** — from any other cwd, `uv run` silently falls back to the
system python and nothing imports.

## Prerequisites

- `uv` (present). All Python deps resolve automatically via `uv run`.
- For replay screenshots only (one-time, ~115 MB):
  `uv tool run --from playwright playwright install chromium`

## Run a game (agent path — start here)

```bash
# fastest smoke: random vs random, ~4s, exit 0 iff all seats DONE
uv run python .claude/skills/run-orbit-wars/driver.py

# real agents, 2P (~10-30s/game), with animated HTML replay
uv run python .claude/skills/run-orbit-wars/driver.py --agents v5,producer --seed 42 --html /tmp/orbit_game.html

# 4P FFA
uv run python .claude/skills/run-orbit-wars/driver.py --agents v5,producer,ow_proto,enders_1000 --seed 7
```

Agent names = anything `agents.load_named_agent` accepts: `v5` (our shipped
fork), `producer`, `ow_proto`, `enders_1000`, `tamrazov_1224`, `distance_1100`,
`reinforce_958`, `shot_validator_hybrid`, `random`. Output: per-seat
status/reward + step count. Specs like `v5:key=val` and `exit:<ckpt>:<config>`
are **arena-only** (see below), not driver/`load_named_agent` names.

## Screenshot a replay

```bash
uv run --with playwright python - <<'EOF'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1280, "height": 900})
    page.goto("file:///tmp/orbit_game.html")
    page.wait_for_timeout(3000)  # player draws via requestAnimationFrame — wait
    page.screenshot(path="/tmp/orbit_replay.png")
    b.close()
EOF
```

Expected: dark board, orange sun at center, numbered planets, colored fleet
arrows, "Step: N" overlay and a playback bar. Blank black board = you
screenshotted before the player drew (keep the 3s wait).

## Arena (THE gate metric)

Round-robin on the real engine, side-alternated paired seeds, resumable CSV
(`outputs/arena/arena.csv` — re-running the same pair/seeds reads cache, bump
`--games` to extend; use `--out /tmp/scratch.csv` for throwaway runs):

```bash
uv run python scripts/arena.py --agents "v5:max_waves_per_turn=4,producer" --games 120 --workers 6
```

⚠️ Measurement rules in CLAUDE.md are hard-won: A/A noise floor ±6% @ n=60
mirror games; never act on n<100.

## Evaluate a checkpoint

```bash
uv run python scripts/eval_fast.py --run v2_exit_producer256_a100 \
    --config configs/v2_exit_producer256.yaml --iters 25 --opponent ow_proto \
    --games 2 --workers 2
```

(n=2 is a smoke; real evals need `--games 60+`. Checkpoints live under
`outputs/checkpoints/<run>/ckpt_NNNNNN.pt`; gitignored, pull from Drive via
`scripts/download_checkpoint.py` if missing.)

## Training smoke (ExIt pipeline, ~15s on CPU)

Real runs happen on Colab (`notebooks/train_colab.ipynb`); locally, verify the
collect→search→distill loop with a micro config — copy `configs/v2_exit.yaml`
and set: `imitation.enabled: false`, `exit.iterations: 1`,
`exit.games_per_iter: 1`, `exit.search_depth: 6`, `exit.search_candidates: 4`,
`exit.dataset_max_iters: 1`, `opponent`/`exit.opponent: ow_proto`,
`eval.eval_every: 999`, scratch `save_dir`/`log_dir`. Then:

```bash
uv run python -m v2.exit_train --config /tmp/v2_exit_smoke.yaml
```

Expected tail: `iter= 1 ... -> checkpoint at iter 1` / `ExIt complete`.

## Build the Kaggle submission bundle

```bash
uv run python scripts/build_v5_bundle.py
```

Builds `outputs/submissions/v5_bundle.tar.gz` AND verifies it by loading the
extracted bundle through Kaggle's real file-agent loader and winning a game vs
random (`verify OK` on the last line). Submit with
`uv run kaggle competitions submit -c orbit-wars -f <tarball> -m "<msg>"`.

## Tests

```bash
uv run python -m scripts.test_gumbel_search       # search invariants + flag-off bit-identity → ALL TESTS PASSED
uv run python -m scripts.test_fast_env_fidelity   # 20 episodes vs real engine → FIDELITY GATE PASSED
```

## Lint / typecheck

```bash
uv run ruff check agents/v5/main.py && uv run pyright agents/v5/main.py
```

Repo-wide `ruff check .` / `uv run pyright` have a known style-only baseline —
fix-as-you-touch (CLAUDE.md); check the files you changed, not the world.

## Gotchas

- **`kaggle_environments` spews ~20 OpenSpiel INFO lines to STDOUT on import**
  (not stderr — `2>/dev/null` won't hide them). Pipe through
  `grep -v open_spiel` to read actual output.
- **`uv run` outside the repo root falls back to system python silently**
  ("No module named ..."): always `cd` to repo root, or pass
  `--project /home/.../orbit_wars`.
- **Run `scripts/*.py` as modules** (`uv run python -m scripts.test_gumbel_search`).
  Running the file path fails `ModuleNotFoundError: No module named 'src'`
  (python sets `sys.path[0]` to `scripts/`) — even where docstrings claim
  otherwise. `arena.py`/`eval_fast.py`/`replay.py` self-patch and work both ways.
- **Firefox cannot screenshot the replay** (two independent failures: snap
  confinement silently loses files under `/tmp`, and `--screenshot` races the
  canvas → blank board with only the playback chrome). Use the playwright
  recipe above.
- **The arena CSV is the shared lab notebook** — A/B history lives in
  `outputs/arena/arena.csv` keyed by (pair, seed). Don't pollute it with smoke
  runs; pass `--out`.
- **`agents/v5` config defaults ARE the shipped agent** (`v5` arena spec =
  current source). Flag experiments default-off; flipping a default changes
  what `scripts/build_v5_bundle.py` ships.
- **Producer-tier agents are torch CPU planners**: ~10–30s per game, and the
  first game in a process pays several seconds of import/JIT warmup.

## Troubleshooting

- `ModuleNotFoundError: No module named 'agents'` from a script you wrote →
  your script's dir is not the repo root; add
  `sys.path.insert(0, <repo_root>)` (the driver does this) or run from root.
- `Firefox is already running, but is not responding` → the user's desktop
  Firefox holds the profile lock; headless needs `--profile <fresh-dir>
  --no-remote` — but see Gotchas: use playwright instead.
- Arena/driver hangs ~30s with no output → normal; you're paying agent
  import + first-game warmup. The OpenSpiel spam printing means it IS running.
