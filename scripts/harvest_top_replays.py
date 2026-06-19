"""Download daily-dataset episodes and keep those featuring a top-K leaderboard
team, for the clone-residual divergence mine. Reuses replay_pulse's throttle-free
daily-dataset path + LB join. Saves kept replays to a flat dir for divergence_mine.

  uv run python scripts/harvest_top_replays.py --date 2026-06-18 --n 300 --top-k 10
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.harvest_teacher import list_day_files_paged  # noqa: E402
from scripts.replay_pulse import _auth, download_episode, list_day_files, load_lb_ratings  # noqa: E402

OUT = Path("/tmp/ow_top")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-06-18")
    ap.add_argument("--n", type=int, default=300, help="episodes to scan")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-keep", type=int, default=120, help="stop once this many kept")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    api = _auth()
    score, rank = load_lb_ratings(api)
    top_teams = {t for t, r in rank.items() if r <= args.top_k}
    print(f"top-{args.top_k} teams: {sorted(top_teams, key=lambda t: rank[t])}")

    files = list_day_files_paged(api, args.date, max_files=max(args.n, 200)) or list_day_files(api, args.date)
    files = files[: args.n]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    kept = 0
    team_seats = Counter()
    for i, nm in enumerate(files):
        rep = download_episode(api, args.date, nm)
        if not rep:
            continue
        names = rep.get("info", {}).get("TeamNames", [])
        hit = [t for t in names if t in top_teams]
        if not hit:
            continue
        # copy the cached file into the flat out dir
        src = REPO / "outputs/replay_pulse/cache" / args.date / nm
        if src.exists():
            shutil.copy(src, out / nm)
        else:
            (out / nm).write_text(json.dumps(rep))
        kept += 1
        for t in hit:
            team_seats[t] += 1
        if kept >= args.min_keep:
            print(f"reached min-keep={args.min_keep} after scanning {i+1}")
            break
        if (i + 1) % 25 == 0:
            print(f"  scanned {i+1}/{len(files)}, kept {kept}")

    print(f"\nkept {kept} replays featuring a top-{args.top_k} team -> {out}")
    print("top-team seat coverage:")
    for t, c in team_seats.most_common():
        print(f"  rank {rank[t]:>2} {t[:28]:<28} {c} games")


if __name__ == "__main__":
    main()
