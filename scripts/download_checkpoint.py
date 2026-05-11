"""Download trained checkpoints from Google Drive via rclone.

Requires rclone configured with a 'gdrive' remote pointing to your Google Drive.
See README / CLAUDE.md for setup instructions.

Usage:
    # Download latest transformer_mixed checkpoint
    uv run python scripts/download_checkpoint.py

    # Download a specific run
    uv run python scripts/download_checkpoint.py --run transformer_dagger

    # Download all checkpoints
    uv run python scripts/download_checkpoint.py --all

    # List available checkpoints on Drive
    uv run python scripts/download_checkpoint.py --list

    # Use a different rclone remote name
    uv run python scripts/download_checkpoint.py --remote mygdrive
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DRIVE_BASE = "orbit_wars_outputs/checkpoints"
LOCAL_BASE = Path("outputs/checkpoints")

DEFAULT_REMOTE = "gdrive"
DEFAULT_RUN = "transformer_mixed"


def _check_rclone() -> None:
    if shutil.which("rclone") is None:
        print("Error: rclone is not installed.")
        print()
        print("Install it:")
        print("  sudo apt install rclone        # Debian/Ubuntu")
        print("  brew install rclone             # macOS")
        print("  curl https://rclone.org/install.sh | sudo bash")
        print()
        print("Then configure Google Drive:")
        print("  rclone config")
        print('  → New remote → name: "gdrive" → type: "drive" → follow prompts')
        sys.exit(1)


def _check_remote(remote: str) -> None:
    result = subprocess.run(
        ["rclone", "listremotes"],
        capture_output=True, text=True,
    )
    remotes = [r.rstrip(":") for r in result.stdout.strip().splitlines()]
    if remote not in remotes:
        print(f"Error: rclone remote '{remote}' not found.")
        print(f"Available remotes: {remotes or '(none)'}")
        print()
        print("Configure it with: rclone config")
        sys.exit(1)


def list_checkpoints(remote: str) -> None:
    print(f"Listing {remote}:{DRIVE_BASE}/")
    subprocess.run(
        ["rclone", "lsf", "--dirs-only", f"{remote}:{DRIVE_BASE}/"],
    )


def download_run(remote: str, run_name: str) -> None:
    src = f"{remote}:{DRIVE_BASE}/{run_name}/ckpt_last.pt"
    dst = LOCAL_BASE / run_name
    dst.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {src} -> {dst}/")
    result = subprocess.run(
        ["rclone", "copy", "--progress", src, str(dst)],
    )
    if result.returncode != 0:
        print(f"Error: rclone copy failed (exit code {result.returncode})")
        sys.exit(1)

    ckpt = dst / "ckpt_last.pt"
    if ckpt.exists():
        size_mb = ckpt.stat().st_size / 1e6
        print(f"Downloaded: {ckpt} ({size_mb:.1f} MB)")
    else:
        print(f"Warning: ckpt_last.pt not found in {dst}")
        print("Available files:")
        for f in sorted(dst.iterdir()):
            print(f"  {f.name}")


def download_all(remote: str) -> None:
    src = f"{remote}:{DRIVE_BASE}/"
    LOCAL_BASE.mkdir(parents=True, exist_ok=True)

    print(f"Downloading all checkpoints: {src} -> {LOCAL_BASE}/")
    result = subprocess.run(
        ["rclone", "copy", "--progress", src, str(LOCAL_BASE)],
    )
    if result.returncode != 0:
        print(f"Error: rclone copy failed (exit code {result.returncode})")
        sys.exit(1)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download trained checkpoints from Google Drive via rclone.",
    )
    parser.add_argument(
        "--run", default=DEFAULT_RUN,
        help=f"Run name to download (default: {DEFAULT_RUN})",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Download all checkpoint runs",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available checkpoint runs on Drive",
    )
    parser.add_argument(
        "--remote", default=DEFAULT_REMOTE,
        help=f"rclone remote name (default: {DEFAULT_REMOTE})",
    )
    args = parser.parse_args()

    _check_rclone()
    _check_remote(args.remote)

    if args.list:
        list_checkpoints(args.remote)
    elif args.all:
        download_all(args.remote)
    else:
        download_run(args.remote, args.run)


if __name__ == "__main__":
    main()
