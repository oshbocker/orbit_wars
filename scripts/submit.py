#!/usr/bin/env python3
"""
Generate a self-contained Kaggle submission file.

Examples
--------
# Submit the deterministic baseline (no model needed)
python scripts/submit.py --baseline

# Submit a trained RL model (weights embedded as base64)
python scripts/submit.py --model outputs/checkpoints/ppo_default_20260501/best_model.zip

# Custom output path
python scripts/submit.py --model outputs/checkpoints/run_a/best_model.zip \\
                          --output outputs/submissions/my_agent.py

# Quickly verify the generated submission runs without error
python scripts/submit.py --model outputs/checkpoints/run_a/best_model.zip --verify
"""

import argparse
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Kaggle-compatible submission.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--model", "-m",
        default=None,
        metavar="PATH",
        help="Path to trained model .zip (RL submission)",
    )
    source.add_argument(
        "--baseline",
        action="store_true",
        help="Generate a baseline (rule-based) submission",
    )

    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="PATH",
        help="Output file path (default: outputs/submissions/submission.py or submission_rl.py)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a 5-step sanity check on the generated submission",
    )
    args = parser.parse_args()

    from agents.rl_agent import export_submission

    if args.baseline:
        out = args.output or "outputs/submissions/submission_baseline.py"
        path = export_submission(None, output_path=out, mode="baseline")
    else:
        out = args.output or "outputs/submissions/submission_rl.py"
        path = export_submission(args.model, output_path=out, mode="rl")

    if args.verify:
        _verify(path)


def _verify(submission_path: Path) -> None:
    """Import the generated submission and run it for a few steps."""
    import importlib.util, sys
    from kaggle_environments import make

    print(f"\nVerifying {submission_path} ...")
    spec = importlib.util.spec_from_file_location("_submission", submission_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    env = make("orbit_wars", debug=False)
    env.reset()
    # Run 5 manual steps to confirm agent doesn't crash
    for _ in range(5):
        obs = env.steps[-1][0].observation
        action = mod.agent(obs)
        assert isinstance(action, list), f"agent() must return a list, got {type(action)}"
        env.step([action, []])

    print("  OK — agent ran 5 steps without errors.")


if __name__ == "__main__":
    main()
