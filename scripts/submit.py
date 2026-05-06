#!/usr/bin/env python3
"""
Generate a self-contained Kaggle submission file, and optionally upload it.

Examples
--------
# Generate aggressive submission (rule-based)
python scripts/submit.py --aggressive

# Generate RL submission with embedded model weights
python scripts/submit.py --model outputs/checkpoints/ppo_default_20260501/best_model.zip

# Generate and upload to Kaggle in one step
python scripts/submit.py --aggressive --upload
python scripts/submit.py --model outputs/checkpoints/run_a/best_model.zip --upload -m "PPO v2 self-play"

# Generate, verify locally, then upload
python scripts/submit.py --model outputs/checkpoints/run_a/best_model.zip --verify --upload

# Custom output path
python scripts/submit.py --model outputs/checkpoints/run_a/best_model.zip \\
                          --output outputs/submissions/my_agent.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

COMPETITION = "orbit-wars"


def main():
    parser = argparse.ArgumentParser(
        description="Generate (and optionally upload) a Kaggle submission.",
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
        "--aggressive",
        action="store_true",
        help="Generate an aggressive (production-rush) submission",
    )
    source.add_argument(
        "--strategic",
        action="store_true",
        help="Generate a strategic (tree-search) submission",
    )

    parser.add_argument(
        "--output", "-o",
        default=None,
        metavar="PATH",
        help="Output file path (default: outputs/submissions/submission_aggressive.py or submission_rl.py)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run a 5-step sanity check on the generated submission before uploading",
    )
    parser.add_argument(
        "--upload", "-u",
        action="store_true",
        help=f"Upload the generated file to Kaggle ({COMPETITION}) via the Kaggle CLI",
    )
    parser.add_argument(
        "--message",
        default=None,
        metavar="MSG",
        help="Submission message shown on Kaggle (default: auto-generated from agent type)",
    )
    args = parser.parse_args()

    from agents.rl_agent import export_submission

    if args.aggressive:
        out = args.output or "outputs/submissions/submission_aggressive.py"
        path = export_submission(None, output_path=out, mode="aggressive")
        default_message = "Aggressive production-rush agent"
    elif args.strategic:
        out = args.output or "outputs/submissions/submission_strategic.py"
        path = export_submission(None, output_path=out, mode="strategic")
        default_message = "Strategic tree-search agent"
    else:
        out = args.output or "outputs/submissions/submission_rl.py"
        path = export_submission(args.model, output_path=out, mode="rl")
        model_label = Path(args.model).parent.name
        default_message = f"RL agent — {model_label}"

    if args.verify:
        _verify(path)

    if args.upload:
        message = args.message or default_message
        _upload(path, message)


def _verify(submission_path: Path) -> None:
    """Import the generated submission and run it for a few steps."""
    import importlib.util
    from kaggle_environments import make

    print(f"\nVerifying {submission_path} ...")
    spec = importlib.util.spec_from_file_location("_submission", submission_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    env = make("orbit_wars", debug=False)
    env.reset()
    for _ in range(5):
        obs = env.steps[-1][0].observation
        action = mod.agent(obs)
        assert isinstance(action, list), f"agent() must return a list, got {type(action)}"
        env.step([action, []])

    print("  OK — agent ran 5 steps without errors.")


def _upload(submission_path: Path, message: str) -> None:
    """Upload submission_path to Kaggle using the Kaggle CLI."""
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print(
            "\nError: 'kaggle' CLI not found.\n"
            "Install it with:  uv sync --extra dev\n"
            "Then configure:   ~/.config/kaggle/kaggle.json  (API key from kaggle.com/settings)\n",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = [
        "kaggle", "competitions", "submit",
        COMPETITION,
        "-f", str(submission_path),
        "-m", message,
    ]
    print(f"\nUploading to Kaggle competition '{COMPETITION}' ...")
    print(f"  file:    {submission_path}")
    print(f"  message: {message}")
    print(f"  command: {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
