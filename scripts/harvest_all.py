#!/usr/bin/env python
"""Harvest ALL available daily-episode datasets into a (Drive-backed) cache — the data
scale-up the Kaggle #30 / vkhydras regime needs (he reached ~47k replays / 28M states this
exact way; we'd been training on ~1500).

Each ``kaggle/orbit-wars-episodes-YYYY-MM-DD`` dataset holds ~4600 episodes, already curated
to the top ~10% of that day's games, and (unlike the throttled ``ListEpisodes`` API) has no
rate limit — so this is download-throughput bound, which is exactly what Colab's bandwidth +
CPUs accelerate. Reuses ``replay_pulse``'s auth / paged-listing / cached-download path; the
only addition is iterating the full date range and a thread pool for download fan-out.

Resumable: ``download_episode`` skips files already on disk, so re-running tops up the cache
(safe across Colab session resets when ``--cache-root`` points at Drive).

    # Local smoke (tiny):
    uv run python scripts/harvest_all.py --start 2026-06-12 --end 2026-06-12 --per-day 5

    # Colab (Drive-backed, full corpus toward ~47k):
    python scripts/harvest_all.py \
        --start 2026-05-19 --per-day 1500 --workers 16 \
        --cache-root /content/drive/MyDrive/orbit_wars/replays
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import scripts.replay_pulse as rp  # noqa: E402
from scripts.harvest_teacher import list_day_files_paged  # noqa: E402

# Earliest known orbit-wars-episodes dataset (competition start window).
DEFAULT_START = "2026-05-19"


def _daterange(start: str, end: str, newest_first: bool = True):
    """Dates in [start, end]. Default newest-first: recent data reflects the CURRENT meta
    (the leaderboard is a red queen — stale games decay in relevance), so when we cap the
    corpus we keep the most leaderboard-relevant days."""
    y0, m0, d0 = (int(x) for x in start.split("-"))
    y1, m1, d1 = (int(x) for x in end.split("-"))
    lo, hi = date(y0, m0, d0), date(y1, m1, d1)
    days = []
    cur = lo
    while cur <= hi:
        days.append(cur.isoformat())
        cur += timedelta(days=1)
    return list(reversed(days)) if newest_first else days


def harvest_day(api, day: str, per_day: int, workers: int) -> tuple[int, int]:
    """Download up to ``per_day`` episodes for ``day``. Returns (downloaded, available)."""
    try:
        files = list_day_files_paged(api, day, max_files=per_day)
    except Exception as e:  # noqa: BLE001 — missing/未published date => skip
        print(f"  {day}: no dataset ({e})")
        return 0, 0
    if not files:
        print(f"  {day}: empty / not published")
        return 0, 0
    files = files[:per_day]
    got = 0
    if workers <= 1:
        for nm in files:
            if rp.download_episode(api, day, nm) is not None:
                got += 1
    else:
        # Each download is a self-contained GET (its own paths); the Kaggle ApiClient
        # tolerates concurrent reads. Fan out the HTTP fetches, which dominate wall-time.
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(rp.download_episode, api, day, nm): nm for nm in files}
            for fut in as_completed(futs):
                try:
                    if fut.result() is not None:
                        got += 1
                except Exception as e:  # noqa: BLE001
                    print(f"    fail {futs[fut]}: {e}")
    print(f"  {day}: {got}/{len(files)} cached", flush=True)
    return got, len(files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=date.today().isoformat(),
                    help="inclusive end date (default: today)")
    ap.add_argument("--per-day", type=int, default=1500,
                    help="max episodes per day (~4600 available; cap to size the corpus)")
    ap.add_argument("--max-total", type=int, default=0,
                    help="stop once this many episodes are cached (0 = no cap); newest days "
                         "first, so the cap keeps the most leaderboard-relevant data")
    ap.add_argument("--oldest-first", action="store_true",
                    help="iterate oldest->newest instead of the default newest->oldest")
    ap.add_argument("--workers", type=int, default=16, help="parallel downloads per day")
    ap.add_argument("--cache-root", default=str(rp.CACHE),
                    help="where to cache replays (point at a Drive path on Colab)")
    args = ap.parse_args()

    # Redirect replay_pulse's cache to the (Drive-backed) root so download_episode +
    # the sharded builder share one location.
    rp.CACHE = Path(args.cache_root)
    rp.CACHE.mkdir(parents=True, exist_ok=True)
    print(f"cache root: {rp.CACHE}")

    api = rp._auth()
    total = avail = 0
    days = _daterange(args.start, args.end, newest_first=not args.oldest_first)
    order = "oldest->newest" if args.oldest_first else "newest->oldest"
    cap = f", stop at {args.max_total} total" if args.max_total else ""
    print(f"harvesting {len(days)} days {args.start}..{args.end} ({order}), "
          f"<= {args.per_day}/day, {args.workers} workers{cap}")
    for day in days:
        # Newest-first + a total cap means we keep the most leaderboard-relevant days.
        budget = args.per_day
        if args.max_total:
            remaining = args.max_total - total
            if remaining <= 0:
                print(f"reached --max-total {args.max_total}; stopping")
                break
            budget = min(budget, remaining)
        g, a = harvest_day(api, day, budget, args.workers)
        total += g
        avail += a
    print(f"\nDONE: cached {total} episodes ({avail} listed) across {len(days)} days "
          f"-> {rp.CACHE}")


if __name__ == "__main__":
    main()
