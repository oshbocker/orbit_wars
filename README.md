# Orbit Wars — RL Agent

Kaggle Orbit Wars competition. Primary goal: learn reinforcement learning and build competitive agents.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the virtual environment and install dependencies:

```bash
uv sync
```

This creates `.venv/` and installs everything from `pyproject.toml`. To also install dev tools (TensorBoard, Jupyter):

```bash
uv sync --extra dev
```

Run any script inside the venv with `uv run`:

```bash
uv run python scripts/evaluate.py
uv run python scripts/train.py
uv run tensorboard --logdir outputs/logs
```

Or activate the venv once for a session:

```bash
source .venv/bin/activate
python scripts/evaluate.py
```

### GPU / CUDA

For CUDA builds of PyTorch, install torch separately before syncing:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
uv sync
```

Python 3.10+ required. GPU is optional — CPU works fine for development.

## Evaluate Agents Locally

The `scripts/evaluate.py` script runs head-to-head matches using the Kaggle environment locally. No GPU or Kaggle account needed.

### Aggressive vs random (quick sanity check)

```bash
python scripts/evaluate.py
```

### Evaluate a trained RL model

When only `--model` is given, aggressive and random are added automatically:

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

### Full matrix (RL models + aggressive + random)

```bash
python scripts/evaluate.py \
    --model outputs/checkpoints/run_a/best_model.zip:rl_v1 \
    --model outputs/checkpoints/run_b/best_model.zip:rl_v2 \
    --aggressive --random \
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
| `--aggressive` | Include the aggressive (production-rush) agent |
| `--random` | Include the random agent |
| `--games N` | Games per matchup (default: 20) |
| `--verbose` | Print per-game results |

## Train an Agent

```bash
# Default: PPO vs aggressive, 500k steps
python scripts/train.py

# Different config
python scripts/train.py --config configs/ppo_selfplay.yaml

# Override specific values
python scripts/train.py --set training.total_timesteps=1000000 env.n_envs=8 training.device=cuda

# Resume from checkpoint
python scripts/train.py --resume outputs/checkpoints/<run>/best_model.zip
```

Checkpoints are saved to `outputs/checkpoints/<run_name>_<timestamp>/`.

### Train on Google Colab (GPU)

Open `notebooks/train_colab.ipynb` in Colab for GPU-accelerated training with persistent Google Drive storage.

1. Upload the notebook to Colab (or open from GitHub)
2. Edit `REPO_URL` in the Setup cell to point to your repo
3. Run Setup — it mounts Drive, clones the repo, and installs deps
4. Edit the Config cell to set your experiment (config file + overrides)
5. Run Training — checkpoints save to `My Drive/orbit_wars_outputs/`

Results persist across Colab sessions via Google Drive. Resume training by setting `resume_from` to a checkpoint path on Drive.

The notebook also supports hyperparameter sweeps (cell 5) and inline TensorBoard monitoring.

### Monitor training

```bash
tensorboard --logdir outputs/logs
```

## Generate and Submit to Kaggle

### One-step: generate + upload

```bash
# Aggressive agent
python scripts/submit.py --aggressive --upload

# Trained RL agent
python scripts/submit.py --model outputs/checkpoints/<run>/best_model.zip --upload

# With a custom submission message
python scripts/submit.py --model outputs/checkpoints/<run>/best_model.zip \
    --upload --message "PPO v2 self-play 2M steps"

# Verify locally before uploading
python scripts/submit.py --model outputs/checkpoints/<run>/best_model.zip --verify --upload
```

### Generate file only (upload manually)

```bash
python scripts/submit.py --aggressive
python scripts/submit.py --model outputs/checkpoints/<run>/best_model.zip
```

Output goes to `outputs/submissions/`. Upload manually with:

```bash
kaggle competitions submit orbit-wars -f outputs/submissions/submission_rl.py -m "my message"
```

### Kaggle CLI setup (one-time)

```bash
uv sync --extra dev   # installs the kaggle package
```

Then add your API credentials — download `kaggle.json` from [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token:

```bash
mkdir -p ~/.config/kaggle
mv ~/Downloads/kaggle.json ~/.config/kaggle/kaggle.json
chmod 600 ~/.config/kaggle/kaggle.json
```

## Repository Structure

```
agents/          # Agent implementations (aggressive.py, rl_agent.py)
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
from agents.aggressive import agent as aggressive

benchmark(my_agent, agent_name="my_agent", n_games=20)
# or
results = run_games(my_agent, aggressive, n_games=50, verbose=True)
```
