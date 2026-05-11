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

### Apex vs random (quick sanity check)

```bash
uv run python scripts/evaluate.py
```

### Evaluate a trained RL model

When only `--model` is given, apex and random are added automatically:

```bash
uv run python scripts/evaluate.py --model outputs/checkpoints/<run>/best_model.zip
```

### Compare two trained models

```bash
puv run ython scripts/evaluate.py \
    --model outputs/checkpoints/run_a/best_model.zip:rl_v1 \
    --model outputs/checkpoints/run_b/best_model.zip:rl_v2 \
    --games 30
```

### Full matrix (RL models + apex + random)

```bash
uv run python scripts/evaluate.py \
    --model outputs/checkpoints/run_a/best_model.zip:rl_v1 \
    --model outputs/checkpoints/run_b/best_model.zip:rl_v2 \
    --apex --random \
    --games 20
```

### Two-model head-to-head only

```bash
uv run python scripts/evaluate.py \
    --model outputs/checkpoints/run_a/best_model.zip \
    --vs    outputs/checkpoints/run_b/best_model.zip \
    --games 50
```

### Options

| Flag | Description |
|------|-------------|
| `--model PATH[:LABEL]` | Add a trained RL model; repeat for multiple |
| `--vs PATH[:LABEL]` | Evaluate `--model` only against this single opponent |
| `--apex` | Include the apex rule-based agent |
| `--hybrid` | Include the hybrid (mission-based) agent |
| `--random` | Include the random agent |
| `--games N` | Games per matchup (default: 20) |
| `--verbose` | Print per-game results |

## Train an Agent

### Transformer PPO (primary pipeline)

```bash
# Train with default config (PPO vs apex, 2000 updates)
uv run python -m src.train --config configs/transformer_ppo.yaml

# DAgger: BC pretrain from hybrid demos + PPO with imitation decay
uv run python -m src.train --config configs/transformer_dagger.yaml
```

Checkpoints are saved to `outputs/checkpoints/<run_name>/` as `.pt` files.

### Legacy SB3 pipeline

```bash
# Default: PPO vs apex, 500k steps
uv run python scripts/train.py

# Different config
uv run python scripts/train.py --config configs/ppo_selfplay.yaml

# Override specific values
uv run python scripts/train.py --set training.total_timesteps=1000000 env.n_envs=8 training.device=cuda

# Resume from checkpoint
uv run python scripts/train.py --resume outputs/checkpoints/<run>/best_model.zip
```

### Train on Google Colab (GPU)

Open `notebooks/train_colab.ipynb` in Colab for GPU-accelerated training with persistent Google Drive storage.

1. Upload the notebook to Colab (or open from GitHub)
2. Edit `REPO_URL` in the Setup cell to point to your repo
3. Run Setup — it mounts Drive, clones the repo, and installs deps
4. Edit the Config cell to set your experiment (config file + overrides)
5. Run Training — checkpoints save to `My Drive/orbit_wars_outputs/`

Results persist across Colab sessions via Google Drive.

### Run a Colab-trained model locally

After training on Colab, download the checkpoint from Google Drive and evaluate locally (no GPU needed):

1. Copy the checkpoint from Drive:
   ```bash
   mkdir -p outputs/checkpoints/transformer_dagger
   cp ~/path/to/Google\ Drive/orbit_wars_outputs/checkpoints/transformer_dagger/ckpt_last.pt \
      outputs/checkpoints/transformer_dagger/ckpt_last.pt
   ```

2. Evaluate against rule-based agents:
   ```bash
   uv run python -c "
   import torch
   from src.config import load_train_config
   from src.policy import TransformerPolicy
   from src.logging import make_eval_agent
   from evaluation.evaluate import run_games, print_results

   cfg = load_train_config('configs/transformer_dagger.yaml')
   device = torch.device('cpu')
   policy = TransformerPolicy(cfg.model, cfg.env).to(device)
   ckpt = torch.load('outputs/checkpoints/transformer_dagger/ckpt_last.pt',
                      map_location=device, weights_only=True)
   policy.load_state_dict(ckpt['policy'])
   policy.eval()

   agent = make_eval_agent(policy, cfg, device)

   from agents.apex import agent as apex
   print_results('rl', 'apex', run_games(agent, apex, n_games=20, verbose=True))

   from agents.hybrid import agent as hybrid
   print_results('rl', 'hybrid', run_games(agent, hybrid, n_games=20, verbose=True))
   "
   ```

3. Play against itself (self-play evaluation):
   ```bash
   uv run python -c "
   import torch
   from src.config import load_train_config
   from src.policy import TransformerPolicy
   from src.logging import make_eval_agent
   from evaluation.evaluate import run_games, print_results

   cfg = load_train_config('configs/transformer_dagger.yaml')
   device = torch.device('cpu')
   policy = TransformerPolicy(cfg.model, cfg.env).to(device)
   ckpt = torch.load('outputs/checkpoints/transformer_dagger/ckpt_last.pt',
                      map_location=device, weights_only=True)
   policy.load_state_dict(ckpt['policy'])
   policy.eval()

   agent = make_eval_agent(policy, cfg, device)
   print_results('rl_p0', 'rl_p1', run_games(agent, agent, n_games=20, verbose=True))
   "
   ```

4. Play a single game with RL against hybrid and save game_replay.html
```
uv run python -c "
import torch
from kaggle_environments import make
from src.config import load_train_config
from src.policy import TransformerPolicy
from src.logging import make_eval_agent
from agents.hybrid import agent as hybrid_agent

cfg = load_train_config('configs/transformer_dagger.yaml')
device = torch.device('cpu')
policy = TransformerPolicy(cfg.model, cfg.env).to(device)
ckpt = torch.load('outputs/checkpoints/transformer_dagger/ckpt_last.pt',
                     map_location=device, weights_only=True)
policy.load_state_dict(ckpt['policy'])
policy.eval()

rl_agent = make_eval_agent(policy, cfg, device)

env = make('orbit_wars', debug=False)
env.run([rl_agent, hybrid_agent])

html = env.render(mode='html', width=800, height=600)
with open('game_replay.html', 'w') as f:
    f.write(html)

reward = env.steps[-1][0].reward
result = 'WIN' if reward and reward > 0 else 'LOSS' if reward and reward < 0 else 'TIE'
print(f'{result} (reward={reward}, {len(env.steps)} steps)')
print('Saved game_replay.html')"
```

The config file must match what was used for training (model architecture, `max_targets`, `ship_fractions`, etc.).

### Monitor training

```bash
uv run tensorboard --logdir outputs/logs
```

## Generate and Submit to Kaggle

### One-step: generate + upload

```bash
# Apex agent
python scripts/submit.py --apex --upload

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
python scripts/submit.py --apex
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
src/             # Transformer PPO pipeline (primary RL effort)
agents/          # Agent implementations (apex.py, hybrid.py, rl_agent.py)
configs/         # YAML experiment configs
envs/            # Gymnasium wrapper for legacy SB3 pipeline
evaluation/      # Core evaluation utilities
notebooks/       # Jupyter notebooks (train_colab.ipynb for Colab GPU training)
scripts/         # CLI entry points (train, evaluate, submit)
training/        # Core PPO training logic (legacy SB3)
outputs/         # Generated files — gitignored
  checkpoints/   # Model .pt / .zip files
  logs/          # TensorBoard event files + CSV metrics
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
from agents.apex import agent as apex

benchmark(my_agent, agent_name="my_agent", n_games=20)
# or
results = run_games(my_agent, apex, n_games=50, verbose=True)
```
