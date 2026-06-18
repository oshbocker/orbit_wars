# New-session prompt — extend contest+detector to 4P FFA (v5.5)

Copy the block below into a fresh Claude Code session.

---

Extend the **contestation overlay + Tier-1 producer detector** (shipped as v5.4 on
2026-06-17, 2P-only) to **4-player FFA**. Full context: `rl_research/CONTESTATION_OVERLAY_FINDINGS.md`
(see the "4P note" at the end) and memory `contestation-overlay`. The overlay lives in
`agents/v5/orbit_lite_v5/contestation.py` + `main.py`; the detector is `_OpponentTracker` +
`_opponent_reactive_status(producer_baseline=, inject_seats=, sources_out=)` in `main.py`,
gated in `run_turn`. In 2P it wins +16.2 vs producer / +11.7 vs producer_v2 with zero
regression vs non-producers. It is **OFF in 4P**: `CONFIG_4P` sets `contest_waves=0`
explicitly. This task turns it on for 4P, safely.

**Why this is the highest-EV next step:** the ladder is 2P + 4P FFA, so ~half our games are
4P and currently get plain v5.3 there. The detector is *already per-seat* — `_OpponentTracker`
tracks every enemy seat's fidelity, and `_opponent_reactive_status` already loops all opponent
seats and accepts an `inject_seats` filter. So per-seat gating ("snipe the producer-clones at
the table, base-play the others") is **structurally already supported**; the work is enabling
it, fitting the time budget, and validating FFA dynamics.

## The hard constraint to solve FIRST: timing

`actTimeout=1s` per step, 60s banked overage; **4P games are already ~4 min** and the
producer-tier planner is 50-100× slower than the retired apex. The overlay runs producer's
**full planner once per opponent seat per turn** (`_opponent_reactive_status` loops seats).
In 2P that's 1 extra planner call/turn; in 4P it's **3** — plus the snipe planning. This may
blow the per-step budget or drain overage. **Before any behavioral work, measure it:** time a
4P turn with `contest_waves=2` vs `=0` (drive `agents/v5/main.py` directly, or add timing to a
4P `scripts/arena.py --players 4` run) and check `remainingOverageTime` never hits 0.

If it's too slow, mitigations (in rough order of preference):
1. **Predict only seats worth predicting** — skip the producer-planner call for a seat whose
   fidelity EMA is already far below threshold AND has enough observations (it won't be sniped
   anyway). Keep predicting near-threshold/under-observed seats. This is the natural per-seat
   short-circuit and likely sufficient.
2. **Lower `contest_opp_waves`** in 4P (fewer predicted waves/seat = cheaper planner).
3. **Cadence** — run detection/prediction every K turns instead of every turn (fidelity EMA
   tolerates gaps; gate state persists between updates).
4. Cap total opponent-planner calls/turn and round-robin seats across turns.

Whatever you pick, **log what was skipped** (the project's no-silent-caps rule) and keep it
default-off / byte-identical when `contest_waves=0`.

## Implement

1. **Recalibrate the gate for 4P.** The 2P thresholds (`contest_fidelity_threshold=0.55`,
   `contest_min_observations=8`, `contest_fidelity_alpha=0.9`) were calibrated on 2P precision
   sequences. 4P boards are denser and opponents launch differently — re-run the calibration
   harness (`/tmp/calib_dump.py` + `/tmp/grid_detect.py` from the v4 detector session, or
   rebuild: dump per-turn source-set **precision** sequences per opponent in 4P, grid over
   alpha/threshold/min_obs). Confirm producer/producer_v2 clear the gate and tamrazov/ow_proto
   /distance_1100/enders_1000 don't. Add 4P-specific knobs to `CONFIG_4P` only if the 2P values
   don't transfer (keep them as overrides, not new global defaults).

2. **Turn the overlay on in 4P, per-seat gated.** Set `CONFIG_4P.contest_waves` > 0 (sweep the
   dose) and make the `run_turn` contest hook pass the gated seat set as `inject_seats` so only
   confirmed-clone seats' captures get sniped. Verify `plan_contestation_waves` and the flip
   detection `(fut_owner != pid) & (fut_owner >= 0)` behave correctly with 3 opponents (they
   should — they're written generic over `player_count`).

3. **Respect FFA dynamics.** In 4P the lesson is "let opponents fight, then strike the winner"
   — sniping a clone's capture can hand tempo to a third player or waste ships in a multi-way
   fight. Consider gating snipes on board position (e.g. only snipe when we're not behind, or
   only the leading clone), and lean on the existing FFA score machinery
   (`ffa_leader_attack_bonus`, nearest-opponent priority) rather than fighting it. Let the gate
   metric decide; don't assume the 2P win transfers.

## Gate (THE metric)

- **Local screen:** `scripts/arena.py --players 4` (memory `arena-4p-mode`) — every 4-agent
  combo, seat-rotated, ranked by reward + final board score. Use a pool mixing clones and
  non-clones, e.g. `v5,v5:contest_waves=0,producer,producer_v2,tamrazov_1224,enders_1000`. The
  critical comparison is **v5(4P contest on) vs v5:contest_waves=0 (plain v5.3 in 4P)** across
  the same combos/seats. Memory warns: **local 4P arena is a SANITY SCREEN only; the ladder is
  the final 4P gate** (the v5.0 4P regression only showed on the ladder). So: pass the local
  screen with no regression, then ship as a ladder A/B and watch the 4P rating.
- Win condition: 4P contest+detector is **≥ plain v5.3 in 4P** (no regression — same strict
  bar as Tier 1), ideally + a measurable edge vs clone-heavy tables. Keep 2P byte-identical to
  v5.4 (don't touch `ProducerLiteConfig` 2P defaults).

## Ship

If it holds, ship per memory `active-two-slot-resubmit-rule`: build with
`scripts/build_v5_bundle.py`, then `uv run kaggle competitions submit orbit-wars -f
outputs/submissions/v5_bundle.tar.gz -m "..."` followed immediately by resubmitting
`producer_bundle.tar.gz`. **Note:** `uv run kaggle ...` authenticates fine even though
`~/.kaggle/kaggle.json` shows `YOUR_KAGGLE_USERNAME` — don't re-doubt it. Kaggle skill ratings
reset per resubmit and climb over hours; judge on the local gate, not the first provisional
score.

---

## Alternatives considered (redirect targets if 4P timing proves intractable)

- **Tier 2 — producer_v2 ensemble (lifts a measured 2P gap).** producer_v2 only cleared the
  gate ~43% of turns (precision 0.58, near threshold) because the base-producer opponent model
  over-predicts captures producer_v2 *declines* (it has reinforce-risk β). Keep a small
  ensemble of opponent models (base producer + producer_v2-with-β) and, per seat, detect/snipe
  with the best-matching model. Should lift producer_v2 retention toward producer's ~1.0,
  capturing more of the +11.7. Cheaper than 4P and self-contained in 2P.
- **Tier 3 — symmetric anti-snipe defense (defensive hedge).** The exploit is symmetric: a
  swarmer/half-drainer can counter-snipe *our* thin captures. Gate a defensive "hold fresh
  captures thicker / small reserve" knob on *low* fidelity (confirmed non-producer) — this
  sidesteps the half-drain tempo-tax that's CLOSED vs producer (Clusters 7/9/12) because we
  only pay the reserve where it pays. Build alongside, ship gated.
