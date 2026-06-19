# Tier 2 — producer_v2 ensemble for the contestation detector (v5.6) — RESUME

> **CLOSED 2026-06-18 — DO NOT RE-RUN THE GATE. Ensemble does NOT ship.**
> Final paired gate n=120: producer_v2 +10.0 ✅, tamrazov/distance no-reg ✅, but
> **plain producer −15.8 win / −833 margin ❌ REGRESSION**. The producer_v2 model mis-fires
> vs plain producer → over-commits; net trades +10 vs producer_v2 for −15.8 vs the dominant
> base = net-negative. thr=0.60 fallback won't help (producer gate already saturated 0.90).
> `contest_ensemble` stays default 0 (v5.4 single-model unchanged). Writeup:
> `rl_research/CONTESTATION_OVERLAY_FINDINGS.md` "Tier 2 ... CLOSED" + memory
> `contestation-overlay`. The notes below are the (now-historical) mid-flight checkpoint.


Checkpoint 2026-06-18 ~12:05. Computer shut down mid-gate. Everything below survives
reboot (working-tree edit to `agents/v5/main.py` + this `outputs/tier2/` dir).
`/tmp` was WIPED — the harness scripts + calibration data were copied here.

## State

### Code — DONE, verified (uncommitted, in working tree)
`agents/v5/main.py` only. Adds Tier-2 ensemble behind `contest_ensemble: int = 0`
(default OFF = byte-identical to v5.4). Changes:
- `contest_ensemble` config knob (ProducerLiteConfig).
- `_OpponentTracker` → per-`(seat, model)` fidelity EMA; `gated_seats` opens on EITHER
  model; new `best_model()` selector; `set_predictions(list_of_dicts)`. `_DETECT_DEBUG`
  tuple is now `(step, seat, model, n_pred, n_obs, n_inter)` (6-tuple, was 4-tuple).
- helpers `_producer_baseline_config()` + `_entries_sources()`.
- `_opponent_reactive_status(..., ensemble, inject_model, sources_out_ensemble)` ensemble
  branch: base producer (β=0) + producer_v2 (β=config.reinforce_size_beta) per seat,
  inject each gated seat with its best model → one merged reactive.
- `run_turn` contest block branches on `contest_ensemble`.

Verified: ruff + pyright clean; bundle builds; **byte-identity** of `contest_ensemble=0`
vs git HEAD confirmed (fixed-obs-sequence replay, `outputs/tier2/byteid_replay.py` — note
it loads HEAD via `agents/v5/_main_head.py`, recreate with `git show HEAD:agents/v5/main.py
> agents/v5/_main_head.py` then delete after).

### Calibration — DONE, strongly positive
`outputs/tier2/calib2p_<opp>.npy`. Grid: `outputs/tier2/grid_2p.py` (EDIT the path: it
reads `/tmp/calib2p_*.npy` → change to `outputs/tier2/calib2p_*.npy`).
Shipped gate (alpha=0.9 thr=0.55 min_obs=8), ensemble ON-fraction:
producer 0.90→0.90, **producer_v2 0.31→0.87 (+0.56)**, tamrazov 0.05→0.19, ow_proto
0.00, distance 0.08→0.16, enders 0.00. Sep clone_min 0.87 / other_max 0.19. thr=0.60 is
the documented fallback if a non-producer regresses (tamrazov→0.08, producer_v2 stays 0.85).

### Gate (paired 2P arena) — PARTIAL (341/960), the ship/close decision
`outputs/tier2/paired_2p_partial.csv` (also `outputs/arena/paired_2p_partial_1204.csv`).
ENS vs SINGLE on identical boards, side-alternated. Analyze with
`uv run python outputs/tier2/analyze_paired_2p.py` (EDIT its CSV path or copy the partial
back to `outputs/arena/paired_2p.csv` first).

- **producer_v2 — COMPLETE n=120: single 44.2% → ens 54.2% win, +10.0 paired (margin
  +109, reward +0.217).** ✅ Target HIT — ensemble recovers the forgone snipes; producer_v2
  now wins >50%, on top of the single-model's prior +11.7 over plain v5.
- producer — n=50 partial: ens 74% vs single 57% (+17) — n<100 NOISE, but no regression.
- **tamrazov_1224, distance_1100 — NOT YET MEASURED (the remaining no-regression risk).**

## To resume

1. (optional) Recreate the byte-id reference if re-verifying: see byteid note above.
2. **Finish the gate for the unmeasured/under-n opponents** (keep the producer_v2 n=120
   result from the partial CSV — do NOT re-run it; `paired_2p.py` opens the CSV with "w"
   and would wipe it). Run a fresh CSV for the remainder:
   ```bash
   cd /home/aeschbacher/git/orbit_wars
   # edit outputs/tier2/paired_2p.py if you want a distinct --out, else it overwrites
   # outputs/arena/paired_2p.csv; the partial is safe in outputs/tier2/.
   uv run python outputs/tier2/paired_2p.py tamrazov_1224,distance_1100,producer 60 41000
   uv run python outputs/tier2/analyze_paired_2p.py   # after pointing it at the new CSV
   ```
   Win condition: tamrazov + distance **no regression** (ENS win% within noise of SINGLE;
   both ~100% saturated), producer no regression at full n. producer_v2 already passed.

## Decision / ship (only if no non-producer regression)
- Flip default `contest_ensemble: int = 0` → `1` in `ProducerLiteConfig` (ships v5.6 ON in
  2P; 4P stays plain v5.3 via `CONFIG_4P.contest_waves=0`).
- `uv run python scripts/build_v5_bundle.py` → `outputs/submissions/v5_bundle.tar.gz` = v5.6.
- **DO NOT submit to Kaggle without user confirmation** (rating-affecting; evicts the older
  active incumbent v5.4 → leaves {v5.3 plain, v5.6}). The submit cmd:
  `uv run kaggle competitions submit orbit-wars -f outputs/submissions/v5_bundle.tar.gz -m "v5.6 contest ensemble"`.
- Update `rl_research/CONTESTATION_OVERLAY_FINDINGS.md` (Tier 2 section) + memory
  `contestation-overlay` with the gate result. If a non-producer regresses → try thr=0.60,
  else keep ensemble default-OFF and document Tier 2 CLOSED (single-model already ships +11.7).

## Cleanup before commit
`rm -f agents/v5/_main_head.py` (if recreated). `outputs/` is gitignored.
The only tracked change to commit is `agents/v5/main.py`.
