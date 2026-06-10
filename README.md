# Orbit Wars — RL Agent

Kaggle Orbit Wars competition. Primary goal: **learn reinforcement learning** and build a
competitive agent. The gate metric is the local arena vs vendored public agents (`scripts/arena.py`); the live effort is
the **v2 Expert Iteration (ExIt)** pipeline (`v2/`) — search → distill on top of an OrbitNet
policy.

> For the full design, game rules, roadmap, and the record of explored-and-abandoned ideas,
> see **`CLAUDE.md`** and **`rl_research/EXPLORED_AND_ABANDONED.md`**.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync                 # creates .venv/ and installs from pyproject.toml
uv sync --extra dev     # also installs TensorBoard / Jupyter
```

Run anything inside the venv with `uv run` (Python 3.10+; GPU optional, CPU is fine for dev):

```bash
uv run python -m v2.exit_train --config configs/v2_exit.yaml
```

For a CUDA build of PyTorch, install torch first: `uv pip install torch --index-url
https://download.pytorch.org/whl/cu121 && uv sync`.

## Train

```bash
# ExIt: BC pretrain from the expert (producer) → collect → search-improve → distill
uv run python -m v2.exit_train --config configs/v2_exit.yaml

# v2 PPO (BC warm start → PPO + mixed self-play) — reference baseline
uv run python -m v2.train --config configs/v2_default.yaml
```

Checkpoints save to `outputs/checkpoints/<run_name>/` as `ckpt_*.pt`. For Colab GPU training
with Drive persistence, use `notebooks/train_colab.ipynb`. After Colab, pull a checkpoint with
`uv run python scripts/download_checkpoint.py` (needs rclone — see CLAUDE.md).

## Evaluate & Replay

```bash
# Fast, side-alternated, paired-seed scorer (the reliable one — high variance, use games>=60)
uv run python scripts/eval_fast.py \
    --run v2_exit_a100 --config configs/v2_exit.yaml --iters 20 --opponent producer --games 60

# Export an HTML replay of a game vs producer
uv run python scripts/replay.py --exit \
    --checkpoint outputs/checkpoints/v2_exit_a100/ckpt_000020.pt \
    --config configs/v2_exit.yaml --opponent producer --seed 42 --output replay.html

# Monitor training
uv run tensorboard --logdir outputs/logs
```

## Submit to Kaggle

The live submission is the ExIt bundle: `v2/agent_v3.py` + `ckpt_last.pt` +
`submission_config.yaml` + the `v2/` and `src/` packages, assembled under
`outputs/submissions/`. The Colab notebook builds this bundle right after training. The
bundled code is verified identical to the live training code.

## Repository Structure

```
v2/          # OrbitNet + ExIt pipeline (PRIMARY) — model, search, exit_train, train, env, ...
src/         # Shared building blocks reused by v2/ (game_types, features, policy, opponents, ...)
agents/      # external/ (vendored public agents) + v5/ (our producer fork, shipped)
configs/     # YAML configs — v2_exit*.yaml are live
evaluation/  # evaluate.py (run_games, head_to_head, print_results)
notebooks/   # train_colab.ipynb (Colab GPU) + explore.ipynb
rl_research/ # STRONGER_EXPERT_SEARCH_PLAN.md (live) + EXPLORED_AND_ABANDONED.md (graveyard)
scripts/     # eval_fast.py, replay.py, download_checkpoint.py, run_embed_ab.py, tests
outputs/     # gitignored — checkpoints/, logs/, submissions/
```

## Agent Interface

Any Kaggle-compatible callable works with the evaluation tools:

```python
def my_agent(obs, config=None) -> list:
    # obs.player   — your player ID (0–3)
    # obs.planets  — [[id, owner, x, y, radius, ships, production], ...]
    # obs.fleets   — [[id, owner, x, y, angle, from_planet_id, ships], ...]
    return [[from_planet_id, angle_radians, num_ships], ...]

# from evaluation.evaluate import run_games, print_results
# from agents import load_named_agent
# print_results("mine", "producer", run_games(my_agent, load_named_agent("producer"), n_games=50, verbose=True))
```
