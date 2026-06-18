#!/usr/bin/env python
"""Harvest a strong NON-producer teacher's (obs, action) pairs from the replay corpus.

Stage 0 of the contested-instrument BC build (`LEADERBOARD_CLIMB_PLAN.md`). The hand-built
half-drainer caps sub-1000 (tempo-bound), so a *contested* non-producer instrument requires
behavior-cloning a real top-tier non-producer agent (Isaiah @ Tufa half-drain / 213tubo
swarm). This script reuses the daily-dataset harvest in ``scripts/replay_pulse.py``.

  # FEASIBILITY (default): which strong non-producer teachers are in the day, and how much
  # data can we extract per team? Gate before any training.
  uv run python scripts/harvest_teacher.py --date 2026-06-12 --n 80

  # EXTRACT: dump one team's (obs[t-1], action[t]) pairs to outputs/teacher_bc/<slug>.jsonl
  uv run python scripts/harvest_teacher.py --date 2026-06-12 --n 400 \
      --extract "Isaiah @ Tufa Labs"

Teacher selection: a seat is a candidate teacher if its team is rated >= --min-rtg AND its
median send-fraction < --max-frac (structurally non-producer = partial sends). The t-1
alignment gotcha (action at steps[t] decided on obs at steps[t-1]) is handled exactly as in
``replay_pulse.fingerprint_seat``. See ``rl_research/TOP_TIER_REPLAY_CORPUS.md``.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.replay_pulse import (  # noqa: E402
    _auth,
    download_episode,
    list_day_files,
    load_lb_ratings,
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "teacher_bc"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:40] or "team"


def list_day_files_paged(api, day: str, max_files: int) -> list[str]:
    """Paginate the daily-dataset listing (the single call caps ~200; nextPageToken walks
    the full ~4600). Returns up to ``max_files`` episode filenames."""
    ref = f"kaggle/orbit-wars-episodes-{day}"
    names: list[str] = []
    token = None
    while len(names) < max_files:
        kwargs = {"page_size": 200}
        if token:
            kwargs["page_token"] = token
        try:
            resp = api.dataset_list_files(ref, **kwargs)
        except TypeError:
            resp = api.dataset_list_files(ref)
        batch = [f.name for f in resp.files]
        if not batch:
            break
        names.extend(batch)
        token = getattr(resp, "nextPageToken", None) or getattr(resp, "next_page_token", None)
        if not token:
            break
    return names[:max_files]


def _seat_records(steps: list, seat: int) -> list[dict]:
    """All (obs[t-1], action[t]) examples for one seat where the actor was ACTIVE.

    Includes idle turns (empty action) — a faithful clone must also learn when to hold.
    Returns [] if the seat is malformed.
    """
    out: list[dict] = []
    for t in range(1, len(steps)):
        if seat >= len(steps[t]) or seat >= len(steps[t - 1]):
            return out
        prev = steps[t - 1][seat]
        if prev.get("status") != "ACTIVE":
            continue
        obs = prev.get("observation")
        if not obs or "planets" not in obs:
            continue
        out.append({"obs": obs, "action": steps[t][seat].get("action") or []})
    return out


def _send_fracs(records: list[dict]) -> list[float]:
    fr: list[float] = []
    for r in records:
        pm = {p[0]: p for p in r["obs"]["planets"]}
        for mv in r["action"]:
            p = pm.get(mv[0])
            if p and p[5] > 0:
                fr.append(mv[2] / p[5])
    return fr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default="2026-06-12")
    ap.add_argument("--n", type=int, default=80, help="episodes to sample")
    ap.add_argument("--min-rtg", type=float, default=1500.0, help="teacher must be rated >= this")
    ap.add_argument("--max-frac", type=float, default=0.75, help="non-producer = median send-frac < this")
    ap.add_argument("--extract", default=None, help="exact team name to dump (obs,action) pairs for")
    args = ap.parse_args()

    api = _auth()
    score, rank = load_lb_ratings(api)
    # paginate so we can pull more than the ~200-file single-page cap (needed to harvest
    # enough games of one rare teacher). --n caps how many we actually download.
    files = list_day_files_paged(api, args.date, max_files=max(args.n, 200))
    if not files:
        files = list_day_files(api, args.date)
    if not files:
        raise SystemExit(f"no files for {args.date}")
    picks = files[: args.n]
    print(f"[{args.date}] downloading {len(picks)} of {len(files)} listed episodes")

    # per-team aggregates
    team_games: dict[str, int] = defaultdict(int)
    team_examples: dict[str, int] = defaultdict(int)   # ACTIVE turns (incl. idle)
    team_actions: dict[str, int] = defaultdict(int)    # turns with >=1 launch
    team_fr: dict[str, list] = defaultdict(list)
    team_wins: dict[str, int] = defaultdict(int)
    team_rtg: dict[str, float] = {}

    extract_records: list[dict] = []
    n_games = 0
    for nm in picks:
        rep = download_episode(api, args.date, nm)
        if not rep:
            continue
        n_games += 1
        names = rep.get("info", {}).get("TeamNames", [])
        rewards = rep.get("rewards", [])
        steps = rep["steps"]
        for seat in range(len(steps[0])):
            team = names[seat] if seat < len(names) else f"seat{seat}"
            recs = _seat_records(steps, seat)
            if not recs:
                continue
            fr = _send_fracs(recs)
            team_games[team] += 1
            team_examples[team] += len(recs)
            team_actions[team] += sum(1 for r in recs if r["action"])
            team_fr[team] += fr
            if seat < len(rewards) and rewards[seat] == 1:
                team_wins[team] += 1
            if team in score:
                team_rtg[team] = score[team]
            if args.extract and team == args.extract:
                for r in recs:
                    extract_records.append({"seat": seat, "episode": nm, **r})

    # rank teachers: strong + non-producer
    rows = []
    for team, n_ex in team_examples.items():
        fr = team_fr[team]
        if not fr:
            continue
        medfrac = st.median(fr)
        rtg = team_rtg.get(team)
        rows.append(
            {
                "team": team, "rtg": rtg, "rank": rank.get(team),
                "games": team_games[team], "wins": team_wins[team],
                "examples": n_ex, "act_turns": team_actions[team],
                "medfrac": medfrac, "n_fr": len(fr),
                "teacher": (rtg is not None and rtg >= args.min_rtg and medfrac < args.max_frac),
            }
        )
    rows.sort(key=lambda r: (r["rtg"] or 0), reverse=True)

    print(f"\n{'team':<26}{'rtg':>6}{'rank':>6}{'games':>6}{'win':>4}{'examples':>9}"
          f"{'act_t':>7}{'medfrac':>8}  teacher?")
    for r in rows:
        if r["rtg"] is None and r["games"] < 2:
            continue
        rt = f"{r['rtg']:.0f}" if r["rtg"] else "—"
        rk = str(r["rank"]) if r["rank"] else "—"
        tag = "  ⭐TEACHER" if r["teacher"] else ""
        print(f"{r['team'][:25]:<26}{rt:>6}{rk:>6}{r['games']:>6}{r['wins']:>4}"
              f"{r['examples']:>9}{r['act_turns']:>7}{r['medfrac']:>8.2f}{tag}")

    teachers = [r for r in rows if r["teacher"]]
    if teachers:
        print(f"\n{len(teachers)} strong non-producer teacher(s) found.")
        for r in teachers:
            per_game = r["examples"] / max(1, r["games"])
            print(f"  {r['team']}: {r['examples']} examples in {r['games']} games "
                  f"(~{per_game:.0f}/game, medfrac {r['medfrac']:.2f}). "
                  f"Full day (~4600 eps) extrapolates to ~{int(per_game * r['games'] / len(picks) * 4600):,} examples "
                  f"if it played a similar share.")
    else:
        print("\n⚠ NO strong non-producer teacher in this sample "
              f"(min_rtg={args.min_rtg}, max_frac={args.max_frac}). Try a different day / larger --n.")

    if args.extract:
        OUT.mkdir(parents=True, exist_ok=True)
        slug = _slug(args.extract)
        dst = OUT / f"{slug}_{args.date}.jsonl"
        with open(dst, "w") as f:
            for r in extract_records:
                f.write(json.dumps(r) + "\n")
        print(f"\nextracted {len(extract_records)} (obs,action) pairs for "
              f"'{args.extract}' -> {dst}")


if __name__ == "__main__":
    main()
