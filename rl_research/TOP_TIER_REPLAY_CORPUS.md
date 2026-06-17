# Top-tier replay corpus — access, schema, findings, daily pulse

*Created 2026-06-14. Diagnostic on real leaderboard replays: is the top-of-ladder
replay corpus worth harvesting for the v2 ExIt value head / BC, and how do the best
agents actually play? Companion memory: `memory/top-tier-replay-diagnostic.md`.*

## TL;DR

- The top of the ladder is **NOT a producer monoculture.** ~⅔ of rated top agents are
  producer-family full-drain clones (corpus from them ≈ what we already have), but **~⅓
  are structurally different** — and the difference is concentrated at the very top:
  **#1 Isaiah @ Tufa Labs (1762) sends ~half a garrison at a time** (median send-fraction
  0.52); **213tubo (1536) fires ~14 fractional waves per turn**. Producer/v5 send the full
  `safe_drain` (≈1.0) in 1–2 waves.
- **Corpus is worth a run — but not whole-pool BC** (dilutes back to producer and averages
  incompatible styles → the BC-clones-below-teacher trap, graveyard Cluster 5). The two
  credible uses: (a) **mine the structural deltas** as gated, mirror-measurable planner
  knobs (the one proven channel — how reinforce-risk won); (b) **value head on real
  top-tier outcomes** = exactly Cluster 10's "one untried variant," but its integration
  ceiling is data-independent, so low-odds.
- **Harvest at scale via the daily datasets, NOT the episode API** (which throttles hard).

## Access (reusable)

### Discover episode IDs — `ListEpisodes` (rate-limited, avoid for bulk)
```
POST https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes
body {"submissionId": N}     # HTTP Basic auth with kaggle.json user/key — NO session cookies
```
Returns every episode that submission played, each with all agents' `submissionId`,
`teamId`, `teamName`, and match-time `initialScore` (so you can snowball-crawl up the
ladder). **BUT it 429s (`RESOURCE_EXHAUSTED`) after a few hundred calls with a
multi-minute+ cooldown** — a 4-hop BFS tripped it and it stayed blocked >10 min. Use only
for small, well-spaced lookups (e.g. the Axis-0 own-episode rating fit), never bulk harvest.

### Pull one replay — `GetEpisodeReplay`
The old `EpisodeService/GetEpisodeReplay` path is now **404**. The live one:
```
GET /api/v1/competitions/episodes/{episode_id}/replay   # via kagglesdk.KaggleClient
```
Basic-auth GET returns 401; you must call through the SDK client (it handles auth):
```python
from kagglesdk import KaggleClient, KaggleEnv
from kagglesdk.competitions.types.competition_api_service import ApiGetEpisodeReplayRequest
req = ApiGetEpisodeReplayRequest(); req.episode_id = EP
with KaggleClient(env=KaggleEnv.PROD) as c:
    rep = c.competitions.competition_api_client.get_episode_replay(req).json()
```
Healthy and fast (~0.5 s, ~4.7 MB/game), separate budget from `ListEpisodes`.

### Bulk harvest — the daily datasets (THE path, no throttle)
- `kaggle/orbit-wars-episodes-YYYY-MM-DD` — **~4600 episodes/day, ~21.5 GB/day,
  ~4.7 MB/episode**, files named `{episodeId}.json`, **individually downloadable**
  (`api.dataset_download_file(ref, name)`) with no observed rate limit. These are already
  curated to the top ~10% of games (per-day **median average agent score ~1350–1530,
  top ~1750–1810** in June).
- `kaggle/orbit-wars-episodes-index` → `manifest.csv` = per-day summary
  (date, episode_count, total_bytes, top_avg_score, median_avg_score) — use it to pick the
  strongest day.
- **Rating join:** the replay JSON's `info.TeamNames` gives the team per seat; join to the
  **full public leaderboard CSV** (`api.competition_leaderboard_download("orbit-wars")` →
  ~4400 teams, name→score→rank). `get_leaderboard()` returns only the top 20.

## Replay schema

Top-level keys: `configuration, description, id, info, module_version, name, rewards,
schema_version, specification, statuses, steps, title, version`.

- `rewards` = final per-seat ±1 (win/loss/tie).
- `info.TeamNames` / `info.Agents` = team identity per seat; `info.seed`.
- `steps` = list over time; **`steps[t]` is a list of N per-agent dicts** each
  `{action, observation, reward, status}`.
- `observation` carries the **FULL board at every step for every seat** (perfect
  information): `planets [id,owner,x,y,radius,ships,production]`,
  `fleets [id,owner,x,y,angle,from_planet_id,ships]`, `comets`, `comet_planet_ids`,
  `angular_velocity`, `initial_planets`, `step`, `remainingOverageTime`.
- `action` = `[[from_planet_id, angle_rad, num_ships], ...]`.
- **CRITICAL alignment gotcha:** the action stored at `steps[t]` was decided on the
  observation at **`steps[t-1]`** (verified: 0 garrison-overdraw violations with t-1 vs
  156 with t). Any value/BC labeling or move-fingerprint MUST use this offset.

## Findings (10 games, 2026-06-09 day, ~1300–1800 tier)

Move-style fingerprint = send-fraction (`num_ships / source garrison`) distribution +
waves/turn. Producer/v5 baseline (our v5.3 + ~1180 opponents): **median 1.00, ≥95% full
`safe_drain`, ~0% fractional, ~1.3–1.6 waves/turn** — the flow-diff signature (one send
size per pair = safe_drain ≈ full drain; speed grows with size so it never wants to
under-send; cf. Cluster 7 "0/81 cheap picks").

Per team (`partial%` = sends below 0.9 of source garrison):

| team | rtg | median frac | partial% | waves/turn | style |
|---|--:|--:|--:|--:|---|
| **Isaiah @ Tufa Labs (#1)** | 1762 | **0.52** | 73% | 1.1 | **different — half-drain** |
| Jake Will (#2) | 1748 | 1.00 | 11% | 1.3 | producer-family |
| Felix M Neumann | 1551 | 1.00 | 29% | 1.0 | different-ish (multi-size) |
| flg | 1541 | 1.00 | 2% | 1.6 | producer-family |
| **213tubo** | 1536 | 0.67 | 64% | **14.2** | **different — fractional swarm** |
| vkhydras / bowwow / Controlvector | 1468–1509 | 1.00 | <10% | ~1.5 | producer-family |
| Audun / DONJYARAHOI (mid, won) | ~1300–1460 | 0.89 | ~52% | 3–4 | different — multi-wave |

In these 10 games the strongly-different agents won 4/4 of their head-to-heads vs
producer-clones — but producer-style Jake Will (#2) also wins, so **different ≠ strictly
better**; the honest read is the top is *mixed* and the single strongest agent is
*different*. (Small n — illustrative, not a measurement.)

## What this does NOT change (graveyard guardrails)

- **Naive fractional/multi-size sends bolted onto producer are already closed** (Cluster 7:
  flow-diff provably prefers full drain; multi-size knob inert, 0/81 cheap picks). Isaiah's
  half-drain is a *different planner*, not producer-with-a-knob — porting it is a structural
  rebuild, not a config flip.
- **Coarse signal second-guessing the exact planner regresses it** (Clusters 6/8/9/10:
  shot-validator, arrival-horizon, defensive-symmetry, value-rerank). A learned value on
  near-ties was INERT even at AUC 0.78.
- The proven win channel is a **gated, mirror-measurable delta that changes what the planner
  MODELS** (reinforce-risk, +150 ladder). New structural ideas must come from somewhere —
  this corpus is now that source.

## Daily pulse (run every day — 10 days left)

`scripts/replay_pulse.py` downloads a sample of the day's curated replays, joins ratings,
fingerprints every seat, classifies producer-family vs different, and appends a trend line
to `outputs/replay_pulse/INDEX.md`.

```bash
uv run python scripts/replay_pulse.py                       # yesterday, 40 games
uv run python scripts/replay_pulse.py --date 2026-06-13 --n 60
uv run python scripts/replay_pulse.py --list-days
```

Watch for: ⭐ rows = agents rated ≥1500 that are structurally different from producer
full-drain. A *new* high-rated different style appearing (or a producer-clone reaching the
very top) is the signal that the meta has moved — the same channel that surfaced
reinforce-risk. Outputs are git-ignored (`outputs/`).
