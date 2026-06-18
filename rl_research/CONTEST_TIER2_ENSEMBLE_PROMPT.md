# New-session prompt — Tier 2: producer_v2 ensemble for the contestation detector (v5.6)

Copy the block below into a fresh Claude Code session.

---

Lift the **contestation overlay's measured 2P gap vs producer_v2** by giving the Tier-1
detector a small **opponent-model ensemble**. Full context: `rl_research/CONTESTATION_OVERLAY_FINDINGS.md`
("Tier 2" in the robustness plan + the "4P CLOSED" section) and memory `contestation-overlay`.
The overlay/detector live in `agents/v5/orbit_lite_v5/contestation.py` + `agents/v5/main.py`
(`_OpponentTracker`, `_opponent_reactive_status`, gated in `run_turn`). This is a **2P, self-
contained** task — it does NOT touch 4P (which stays plain v5.3; the 4P contest extension is
CLOSED — snipe value doesn't transfer to FFA).

## The measured gap

The overlay snipes the planets a producer-family opponent will capture-thin, gated by a
per-seat **source-set precision EMA** against ONE opponent model: the **base producer** flow-
diff (`_opponent_reactive_status(producer_baseline=True, ...)` at `main.py:900`, which strips
every v5 knob — crucially `reinforce_size_beta=0.0` at `main.py:909`). Gate (2P, paired,
`scripts/arena.py`): vs **producer +16.2** (gate ON ~100% of turns, precision ~0.99) but vs
**producer_v2 only +11.7** (gate ON ~43% of turns, precision ~0.58, near the 0.55 threshold).

**Why producer_v2 under-retains:** producer_v2 = producer + the slawekbiel V2 ETA-aware
reinforcement-risk capture floor (our `reinforce_size_beta=2.2` knob, `main.py:154`), so it
**declines** captures the base-producer model predicts it will make (captures the enemy can
reinforce mid-flight). The base model therefore **over-predicts** producer_v2's launch sources
→ precision drops → the gate sits near-threshold → the overlay is OFF ~57% of producer_v2
turns, forgoing snipes worth a chunk of the missing +4.5 (the gap to producer's +16.2).

## The idea (ensemble detection + per-seat best-model sniping)

Maintain a SMALL ensemble of opponent models, score each enemy seat's fidelity under EACH, and
detect/snipe each seat with its **best-matching** model:

1. **base-producer** model — `reinforce_size_beta=0.0` (the current one; matches plain producer).
2. **producer_v2** model — `reinforce_size_beta=2.2` + `reinforce_eta_free=3.0` /
   `reinforce_eta_scale=12.0` (the V2 capture floor kept ON; matches producer_v2's declines).

Per seat, the v2 model should predict producer_v2's *actual* (reduced) source set → precision
toward ~1.0 → gate ON far more often → more snipes → recover the forgone +4.5. Plain producer
keeps scoring highest under the base model. A seat is gated ON if EITHER model clears the bar,
and is **sniped using the reactive projection from its best-matching model** (the model whose
EMA precision is higher for that seat).

## Implement

- **Two opponent configs.** In `_opponent_reactive_status` (`main.py:~872`), the
  `producer_baseline` branch builds one `opp_config`. Generalize to build (or accept) the model
  variant: base = current strip; v2 = same strip but KEEP `reinforce_size_beta` (+ the two
  `reinforce_eta_*`). Both are cheap re-runs of `plan_lite_waves` with a different `opp_config`
  — measure the added cost (it's a 2nd planner pass/seat; 2P so 1 extra seat, should be well
  under the 1s budget, but confirm `remainingOverageTime` like the 4P timing check did).
- **Per-model fidelity in `_OpponentTracker`** (`main.py:~800`). Today it stores one
  `pred_sources[o]` and one `fid[o]`. Make these per-model: `pred_sources[(o, model)]`,
  `fid[(o, model)]`, `turns_observed[(o, model)]`. `observe()` scores precision for each model's
  prediction against the seat's actual launches; `set_predictions()` stores both models' source
  sets; `gated_seats()` returns seats where `max_model fid >= threshold` (with min_obs), AND
  expose which model won per seat so the injection can pick it.
- **Per-seat model selection in the injection.** Currently `run_turn` (`main.py:~1085`) calls
  `_opponent_reactive_status(..., producer_baseline=True, inject_seats=gated)` — ONE model for
  all seats. Tier 2 must inject each gated seat using ITS best model. Either run the projection
  twice (once per model, each injecting only the seats that prefer it) and merge, or thread a
  per-seat model map into `_opponent_reactive_status`. Keep it **side-effect-free** (the
  snapshot/restore around `record_fleet_arrivals` at `main.py:~960`).
- **Default-OFF / byte-identical.** Gate the whole ensemble behind a new knob (e.g.
  `contest_ensemble: int = 0`); when 0 the detector is exactly today's single base-producer
  model → v5.4 byte-identical. 4P untouched (`CONFIG_4P.contest_waves=0`).

## Gate (THE metric — `scripts/arena.py`, 2P side-alternated paired seeds, n>=100)

Re-run the contest gate with the ensemble ON vs the current single-model detector:
- **vs producer_v2**: target — lift from +11.7 toward producer's +16.2 (the whole point).
- **vs producer**: no regression (base model must still win; ensemble shouldn't dilute it).
- **vs non-producers** (tamrazov_1224, ow_proto, distance_1100, enders_1000): **no regression**
  — the v2 model adds a SECOND chance to false-positive, so RE-VALIDATE the OFF-bias. Rebuild
  the precision-sequence calibration (dump per-turn per-model precision, grid alpha/threshold/
  min_obs) and confirm both models stay <threshold for non-producers. Harnesses to adapt: the
  4P calibration `/tmp/calib_4p_one.py` + `/tmp/grid_4p.py` (set 2P, add the 2nd model to the
  `_DETECT_DEBUG` tuple), and the paired comparison `/tmp/paired_4p.py` (set 2P). Remember the
  **n<100 mirror noise floor** — don't act on small-n; the producer_v2 lift must clear it.

Win condition: contest+ensemble **>= contest-single-model everywhere**, with a measurable
producer_v2 lift. Keep producer + non-producers within noise. 2P only; 4P byte-identical.

## Ship

Per memory `active-two-slot-resubmit-rule`, current active-2 = {v5.4 contestation, v5.3 plain}.
If the ensemble holds, it becomes **v5.6** (supersedes v5.4 in the contest slot). Build both
bundles with `scripts/build_v5_bundle.py` (v5.6 = `contest_ensemble=1` default; the v5.3 plain
slot is `contest_waves=0`), then submit so the latest-2 are {v5.6, v5.3} — submit the slot you
want to KEEP last (a new submission evicts the older active incumbent). `uv run kaggle
competitions submit orbit-wars -f ...` authenticates despite the `YOUR_KAGGLE_USERNAME`
placeholder. Ratings reset per (re)submit and climb over hours — judge on the local gate.

## Alternatives / redirects

- If the producer_v2 lift doesn't clear the noise floor, the ensemble is dead weight — keep it
  default-OFF and document (Tier 2 CLOSED). The single-model detector already ships the +11.7.
- **Tier 3 — symmetric anti-snipe defense** (the other deferred tier): the exploit is symmetric;
  a swarmer/half-drainer can counter-snipe OUR thin captures. Gate a defensive "hold fresh
  captures thicker / small reserve" knob on CONFIRMED-LOW fidelity (non-producer) seats — this
  sidesteps the half-drain tempo-tax that's CLOSED vs producer (Clusters 7/9/12) because the
  reserve is only paid vs agents that actually punish thin captures.
