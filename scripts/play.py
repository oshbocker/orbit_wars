#!/usr/bin/env python3
"""
Play Orbit Wars interactively in the browser.

Launch a local Flask server that renders the game board on a canvas.
You play as Player 0; the opponent (default: aggressive agent) plays
automatically on the server side.

Usage:
    python scripts/play.py                 # default: port 5000, aggressive opponent
    python scripts/play.py --port 8080     # custom port
    python scripts/play.py --opponent random
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_root))

from play.app import create_app


def main():
    parser = argparse.ArgumentParser(
        description="Play Orbit Wars interactively in the browser.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--port", type=int, default=5000, help="Port to serve on (default: 5000)")
    parser.add_argument("--opponent", default="aggressive", choices=["aggressive", "random"],
                        help="Opponent agent type (default: aggressive)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    app = create_app(opponent=args.opponent)

    print()
    print("=" * 50)
    print(f"  Open http://127.0.0.1:{args.port} in your browser to play!")
    print("=" * 50)
    print()

    app.run(host="127.0.0.1", port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
