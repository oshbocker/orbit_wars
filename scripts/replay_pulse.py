#!/usr/bin/env python
"""Daily leaderboard-replay pulse for Orbit Wars.

Downloads a sample of that day's curated top-tier replays from the public
`kaggle/orbit-wars-episodes-YYYY-MM-DD` dataset, joins each seat to the public
leaderboard rating (via the replay's `info.TeamNames`), fingerprints each
agent's move style (full-drain "producer-family" vs fractional/multi-wave
"different"), and writes a dated markdown pulse + appends a one-line trend to
an index. Run it once a day to keep a finger on the meta.

    uv run python scripts/replay_pulse.py                 # yesterday, 40 games
    uv run python scripts/replay_pulse.py --date 2026-06-13 --n 60
    uv run python scripts/replay_pulse.py --list-days     # show available dates

Why the dataset and not the episode API: `ListEpisodes` (the /api/i/ endpoint)
429s hard after a few hundred calls; the daily datasets have no such throttle
and are already curated to the top ~10% of games. See
`rl_research/TOP_TIER_REPLAY_CORPUS.md`.

Requires kaggle.json (username/key). Writes under outputs/replay_pulse/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import zipfile
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "replay_pulse"
CACHE = OUT / "cache"


def _auth():
    creds = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]
    from kaggle import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def load_lb_ratings(api) -> tuple[dict[str, float], dict[str, int]]:
    """name -> score, name -> rank, from the full public leaderboard CSV (~4400 teams)."""
    lbdir = CACHE / "lb"
    lbdir.mkdir(parents=True, exist_ok=True)
    api.competition_leaderboard_download("orbit-wars", path=str(lbdir))
    for z in glob.glob(str(lbdir / "*.zip")):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(lbdir)
    csvs = sorted(glob.glob(str(lbdir / "*.csv")))
    import csv as _csv

    score, rank = {}, {}
    with open(csvs[-1], encoding="utf-8-sig") as f:
        for row in _csv.DictReader(f):
            try:
                score[row["TeamName"]] = float(row["Score"])
                rank[row["TeamName"]] = int(row["Rank"])
            except (ValueError, KeyError):
                pass
    return score, rank


def list_day_files(api, day: str) -> list[str]:
    ref = f"kaggle/orbit-wars-episodes-{day}"
    try:
        resp = api.dataset_list_files(ref, page_size=400)
    except TypeError:
        resp = api.dataset_list_files(ref)
    return [f.name for f in resp.files]


def download_episode(api, day: str, name: str) -> dict | None:
    ref = f"kaggle/orbit-wars-episodes-{day}"
    dst = CACHE / day
    dst.mkdir(parents=True, exist_ok=True)
    path = dst / name
    if not path.exists():
        try:
            api.dataset_download_file(ref, name, path=str(dst))
        except Exception as e:  # noqa: BLE001
            print(f"  download fail {name}: {e}")
            return None
        if (dst / (name + ".zip")).exists():
            with zipfile.ZipFile(dst / (name + ".zip")) as z:
                z.extractall(dst)
    if not path.exists():
        return None
    return json.load(open(path))


def fingerprint_seat(steps: list, seat: int) -> dict:
    """Send-fraction (sent/source garrison) + wave stats for one seat.

    NOTE: the action stored at steps[t] was decided on the observation at
    steps[t-1] (verified: 0 garrison violations vs 156 if aligned to t).
    """
    fr, waves = [], []
    idle = tot = 0
    for t in range(1, len(steps)):
        if seat >= len(steps[t]):
            return {}
        act = steps[t][seat].get("action") or []
        prev = steps[t - 1][seat]
        if prev.get("status") != "ACTIVE":
            continue
        tot += 1
        if not act:
            idle += 1
            continue
        pm = {p[0]: p for p in prev["observation"]["planets"]}
        waves.append(len(act))
        for mv in act:
            p = pm.get(mv[0])
            if p and p[5] > 0:
                fr.append(mv[2] / p[5])
    return {"fr": fr, "waves": waves, "idle": idle, "tot": tot}


def classify(fr: list, waves: list) -> str:
    """Producer-family = full-drain safe_drain (≈no fractional sends, few waves)."""
    if not fr:
        return "idle"
    pct_partial = 100 * sum(1 for f in fr if f < 0.9) / len(fr)
    wv = st.mean(waves) if waves else 0
    if pct_partial < 15 and wv < 3:
        return "producer"
    return "different"


def run(day: str, n: int) -> Path:
    api = _auth()
    score, rank = load_lb_ratings(api)
    files = list_day_files(api, day)
    if not files:
        raise SystemExit(f"no files for {day} (dataset may not be published yet)")
    # sample n evenly across the listing
    step = max(1, len(files) // n)
    picks = files[::step][:n]

    team_fr: dict[str, list] = defaultdict(list)
    team_w: dict[str, list] = defaultdict(list)
    team_games: dict[str, int] = defaultdict(int)
    team_wins: dict[str, int] = defaultdict(int)
    n_games = 0
    for nm in picks:
        rep = download_episode(api, day, nm)
        if not rep:
            continue
        n_games += 1
        names = rep.get("info", {}).get("TeamNames", [])
        rewards = rep.get("rewards", [])
        steps = rep["steps"]
        for seat in range(len(steps[0])):
            team = names[seat] if seat < len(names) else f"seat{seat}"
            fp = fingerprint_seat(steps, seat)
            if not fp or not fp["fr"]:
                continue
            team_fr[team] += fp["fr"]
            team_w[team] += fp["waves"]
            team_games[team] += 1
            if seat < len(rewards) and rewards[seat] == 1:
                team_wins[team] += 1

    rows = []
    for team, fr in team_fr.items():
        if not fr:
            continue
        wv = st.mean(team_w[team]) if team_w[team] else 0.0
        pct_partial = 100 * sum(1 for f in fr if f < 0.9) / len(fr)
        rows.append(
            {
                "team": team,
                "rtg": score.get(team),
                "rank": rank.get(team),
                "games": team_games[team],
                "wins": team_wins[team],
                "n": len(fr),
                "medfrac": st.median(fr),
                "pct_partial": pct_partial,
                "waves": wv,
                "style": classify(fr, team_w[team]),
            }
        )
    rows.sort(key=lambda r: (r["rtg"] is not None, r["rtg"] or 0), reverse=True)

    OUT.mkdir(parents=True, exist_ok=True)
    report = OUT / f"{day}.md"
    rated = [r for r in rows if r["rtg"]]
    hi = [r for r in rated if r["rtg"] >= 1500]
    hi_diff = [r for r in hi if r["style"] == "different"]
    with open(report, "w") as f:
        f.write(f"# Replay pulse — {day}\n\n")
        f.write(f"Games sampled: {n_games}.  Distinct teams: {len(rows)}.  ")
        f.write(f"Rated ≥1500 seats: {len(hi)} ({len(hi_diff)} non-producer-style).\n\n")
        f.write("`partial%` = sends below 0.9 of source garrison; producer flow-diff ≈ full drain (≈0%).\n\n")
        f.write("| rank | team | rtg | g | win | medfrac | partial% | waves | style |\n")
        f.write("|--:|---|--:|--:|--:|--:|--:|--:|---|\n")
        for r in rows:
            if r["rtg"] is None and r["games"] < 1:
                continue
            rk = r["rank"] if r["rank"] else ""
            rt = f"{r['rtg']:.0f}" if r["rtg"] else "—"
            star = " ⭐" if r["style"] == "different" and r["rtg"] and r["rtg"] >= 1500 else ""
            f.write(
                f"| {rk} | {r['team'][:24]} | {rt} | {r['games']} | {r['wins']} | "
                f"{r['medfrac']:.2f} | {r['pct_partial']:.0f}% | {r['waves']:.1f} | {r['style']}{star} |\n"
            )
        f.write("\n⭐ = rated ≥1500 AND structurally different from producer full-drain (meta-watch).\n")

    # append trend line to index
    index = OUT / "INDEX.md"
    new = not index.exists()
    with open(index, "a") as f:
        if new:
            f.write("# Replay pulse — daily trend\n\n")
            f.write("| date | games | rated≥1500 | non-producer≥1500 | top different agents |\n")
            f.write("|---|--:|--:|--:|---|\n")
        watch = ", ".join(
            f"{r['team'][:16]}({r['rtg']:.0f},f{r['medfrac']:.2f},w{r['waves']:.0f})" for r in hi_diff[:4]
        )
        f.write(f"| {day} | {n_games} | {len(hi)} | {len(hi_diff)} | {watch} |\n")

    print(f"wrote {report}")
    print(f"  rated≥1500 seats: {len(hi)} | non-producer-style among them: {len(hi_diff)}")
    for r in hi_diff:
        print(f"    ⭐ {r['team']} ({r['rtg']:.0f}) medfrac={r['medfrac']:.2f} waves={r['waves']:.1f}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=(date.today() - timedelta(days=1)).isoformat(),
                    help="YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--n", type=int, default=40, help="games to sample (default 40)")
    ap.add_argument("--list-days", action="store_true", help="list available daily datasets and exit")
    args = ap.parse_args()
    if args.list_days:
        api = _auth()
        ds = api.dataset_list(search="orbit-wars-episodes-2026")
        for ref in sorted({d.ref for d in ds}, reverse=True)[:14]:
            print(ref)
        return
    run(args.date, args.n)


if __name__ == "__main__":
    main()
