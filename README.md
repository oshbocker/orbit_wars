# Orbit Wars — RL Agent

Kaggle Orbit Wars competition. Primary goal: learn reinforcement learning and build competitive agents.

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended. GPU optional (CPU training works fine for development).

## Evaluate Agents Locally

The `scripts/evaluate.py` script runs head-to-head matches using the Kaggle environment locally. No GPU or Kaggle account needed.

### Baseline vs random (quick sanity check)

```bash
python scripts/evaluate.py
```

### Evaluate a trained RL model

When only `--model` is given, baseline and random are added automatically:

```bash
python scripts/evaluate.py --model outputs/checkpoints/<run>/best_model.zip
```

### Compare two trained models

```bash
python scripts/evaluate.py \
    --model outputs/checkpoints/run_a/best_model.zip:rl_v1 \
    --model outputs/checkpoints/run_b/best_model.zip:rl_v2 \
    --games 30
```

### Full matrix (RL models + baseline + random)

```bash
python scripts/evaluate.py \
    --model outputs/checkpoints/run_a/best_model.zip:rl_v1 \
    --model outputs/checkpoints/run_b/best_model.zip:rl_v2 \
    --baseline --random \
    --games 20
```

### Two-model head-to-head only

```bash
python scripts/evaluate.py \
    --model outputs/checkpoints/run_a/best_model.zip \
    --vs    outputs/checkpoints/run_b/best_model.zip \
    --games 50
```

### Options

| Flag | Description |
|------|-------------|
| `--model PATH[:LABEL]` | Add a trained RL model; repeat for multiple |
| `--vs PATH[:LABEL]` | Evaluate `--model` only against this single opponent |
| `--baseline` | Include the deterministic rule-based agent |
| `--random` | Include the random agent |
| `--games N` | Games per matchup (default: 20) |
| `--verbose` | Print per-game results |

## Train an Agent

```bash
# Default: PPO vs baseline, 500k steps
python scripts/train.py

# Different config
python scripts/train.py --config configs/ppo_selfplay.yaml

# Override specific values
python scripts/train.py --set training.total_timesteps=1000000 env.n_envs=8 training.device=cuda

# Resume from checkpoint
python scripts/train.py --resume outputs/checkpoints/<run>/best_model.zip
```

Checkpoints are saved to `outputs/checkpoints/<run_name>_<timestamp>/`.

### Monitor training

```bash
tensorboard --logdir outputs/logs
```

## Generate a Kaggle Submission

```bash
# Baseline (deterministic rule-based)
python scripts/submit.py --baseline

# Trained RL model with embedded weights
python scripts/submit.py --model outputs/checkpoints/<run>/best_model.zip

# Also verify the submission runs 5 steps
python scripts/submit.py --model outputs/checkpoints/<run>/best_model.zip --verify
```

Output goes to `outputs/submissions/submission.py` — upload this file directly to Kaggle.

## Repository Structure

```
agents/          # Agent implementations (baseline.py, rl_agent.py)
configs/         # YAML experiment configs
envs/            # Gymnasium wrapper (observation + action encoding)
evaluation/      # Core evaluation utilities
notebooks/       # Jupyter notebooks for exploration
scripts/         # CLI entry points (train, evaluate, submit)
training/        # Core PPO training logic
outputs/         # Generated files — gitignored
  checkpoints/   # Model .zip files
  logs/          # TensorBoard event files
  submissions/   # Generated submission.py files
```

## Agent Interface

Any Kaggle-compatible agent callable works with the evaluation tools:

```python
def my_agent(obs, config=None) -> list:
    # obs.player       — your player ID (0–3)
    # obs.planets      — [[id, owner, x, y, radius, ships, production], ...]
    # obs.fleets       — [[id, owner, x, y, angle, from_planet_id, ships], ...]
    return [[from_planet_id, angle_radians, num_ships], ...]
```

Use it directly in a script:

```python
from evaluation.evaluate import run_games, benchmark
from agents.baseline import agent as baseline

benchmark(my_agent, agent_name="my_agent", n_games=20)
# or
results = run_games(my_agent, baseline, n_games=50, verbose=True)
```
