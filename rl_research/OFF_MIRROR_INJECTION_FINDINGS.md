# Off-mirror opponent-injection gate — findings (2026-06-16)

**Question.** Cluster 11 (`EXPLORED_AND_ABANDONED.md`) tested v5's opponent-injection
hook (`opp_inject_waves`, `agents/v5/main.py:_opponent_reactive_status`) only as a
**mirror** A/B (`v5:opp_inject_waves=N` vs `v5`) and it went **INERT** (waves 1/3/6 =
46.7/57.5/56.2% vs a 55.4% A/A floor). Root cause: the mirror can only measure
level-1-vs-level-0 exploitability of a *producer self-model*, and the flow-diff scorer
(`Δnet_me − ΣΔnet_opp`) already prices producer-like opponents in. Neither the mirror nor
the public pool (which ceilings at ~99% vs producer-tier) can see whether *anticipating a
genuinely non-producer opponent* buys anything. The escape hypothesis for the learned
opponent-prediction head was: a head modeling the ~⅓ of the top tier that is **not**
producer-style (half-drainers like Isaiah @ Tufa #1/1762, swarmers like 213tubo/1536 —
see `TOP_TIER_REPLAY_CORPUS.md`) could convert — *but only if we can measure it
off-mirror first.* This is the instrument that question needs. **Measurement build only;
no learned model.**

## What was built

**Non-producer archetype FIXTURES** (`agents/external/`, fresh-loaded via
`agents.load_named_agent`; registered in `agents/external/__init__._FILE_AGENTS`):
- `half_drainer.py` — Isaiah-style. One cheap capture per source per turn (~1.4
  waves/turn), keeps reserves (`drain_cap=0.6`).
- `swarmer.py` — 213tubo-style. Each capture is emitted as `split_k=4` equal
  same-target sub-fleets (identical source/angle/size → identical speed → same arrival →
  they aggregate in combat, so the capture still succeeds) across up to 4 targets →
  **~11 waves/turn measured** (vs half_drainer's ~3.7), the genuine swarm signature
  without the "many tiny fleets can't capture anything" failure of naive swarming.
- `archetype_common.py` — shared deterministic planner. The **non-producer signature is
  capture-minimal sizing**: producer/v5 *full-drain* (≥95% of sends are the whole
  `safe_drain` garrison), whereas these send only enough to take a target and keep the
  rest in reserve (partial sends — the structural delta a producer self-model cannot
  predict). Robust `Struct→dict→list` obs parsing, sun avoidance, orbiting-target
  leading. Params seeded from the replay diagnostic; capture-aware sizing is also what
  makes them credible (a fixed-fraction send can never capture a 9–27-ship neutral from a
  10-ship home → never expands).

Fixtures verified: both load and play full games; they expand coherently (peak ~30
planets), beat **each other** ~50/50, and are behaviourally distinct in the intended
dimension (swarmer ~11 waves/turn vs half_drainer ~3.7) — genuinely competing, distinct
non-producer styles, not degenerate.

**The off-mirror gate** (`scripts/off_mirror_gate.py`, built on `scripts/arena.py`'s real
Kaggle env + `_build_agent`/`_final_scores`): `v5:opp_inject_waves=N` vs `v5` (base,
N=0), **both vs the same archetype** (off-mirror, not a mirror), side-alternated on
**paired seeds**, resumable CSVs in `outputs/arena/offmirror_<archetype>.csv`. Dose curve
N ∈ {1,3,6} like Cluster 11. **n=120 per cell** (the project floor).

## The measurement problem, and the instrument it forced

Producer-tier v5 beats the hand-built fixtures **100%** on win-rate — the *same ~99%
ceiling the public pool hits*. A pure win-rate gate is therefore **blind off-mirror too.**
So, exactly as the task anticipated ("margin asymmetry was the only signal Cluster 11
saw"), the **primary instrument is margin**, measured paired-by-seed (same map for ON and
OFF removes per-map variance):
- **steps-to-elimination** — v5 fully eliminates the archetype (`score_arch=0` in 100% of
  games), so steps = close-out speed. This is *not* saturated (steps vary widely, tight
  paired SE ≈ ±6) → a sensitive instrument even under win-rate saturation.
- final score margin (noisier — dominated by how much v5 banks).

## Results (n=120 per cell, side-alternated paired seeds; all 480/480 DONE, 0 errors)

Win-rate, both archetypes, **every** dose: **100.0%** (Wilson 95% CI [97%,100%];
120/120, zero archetype wins). Saturated → uninformative, as expected.

Paired Δ vs inject-OFF (same seed+side; **+Δ = injection better**; `*` = >1.96 SE from 0):

| archetype | dose | Δ steps-faster | Δ score-margin | Δ win% | signif? |
|-----------|------|---------------:|---------------:|-------:|:-------:|
| half_drainer | waves=1 | −3.9 (±5.8) | +386.9 (±455.8) | +0.0% | no |
| half_drainer | waves=3 | −3.4 (±8.3) | +220.9 (±450.2) | +0.0% | no |
| half_drainer | waves=6 | −1.4 (±5.7) | +102.2 (±420.1) | +0.0% | no |
| swarmer | waves=1 | +1.5 (±8.7) | +127.7 (±484.1) | +0.0% | no |
| swarmer | waves=3 | +1.8 (±7.7) | −200.2 (±533.2) | +0.0% | no |
| swarmer | waves=6 | +6.1 (±7.6) | +109.6 (±525.1) | +0.0% | no |

**Floor context.** Mirror A/A floor (Cluster 11) = 55.4%; the Cluster 11 mirror itself
saw a real-but-insufficient ~10-step margin asymmetry. Here the off-mirror inject-OFF
baseline win-rate = 100% (no win-rate headroom), and the margin noise floor is paired SE
≈ ±6 steps / ±450 score.

## Verdict — the signal does NOT clear noise. FOLD the learned-head track.

**No instrument clears the bar, on either archetype, at any dose.** Every paired
steps-faster Δ ∈ [−3.9, +6.1] sits inside its ±5.7–8.7 SE (and is negative as often as
positive — injection is, if anything, marginally **slower** vs half_drainer; the largest,
swarmer waves=6 at +6.1, still falls short of its ±7.6 bar and is not monotonic with the
saturated 100% win-rate). Every score-margin Δ sits inside its ±420–533 SE and is
**non-monotonic / sign-flipping** in dose (half_drainer 387→221→102 *decreasing*; swarmer
+128→−200→+110 *sign-flipping*) — the fingerprint of noise, not a dose-response. Win-rate
is flat 100%.

This is **stronger than the Cluster 11 mirror result, and points the same way.** The
margin instrument here is *not* saturated (steps vary, SE tight) yet reads ≈0 — even
smaller than the ~10-step asymmetry the mirror saw. **The Cluster 11 redundancy finding
generalizes off-mirror:** the flow-diff scorer's pricing of opponent value is robust
*beyond* producer-style opponents — exact 1-ply injection of a non-producer opponent's
own best-response buys nothing measurable. If the *exact* opponent model is inert
off-mirror, a *learned* (approximate) head of those opponents is very unlikely to convert.

**→ Fold the learned opponent-prediction-head track at current EV.** `opp_inject_waves`
stays default 0 (shipped v5 unchanged). The competition EV remains where
`LEADERBOARD_CLIMB_PLAN.md` / the big-swings direction put it: structural deltas +
meta-monitoring, not opponent-modeling on a locally-optimal flow-diff base.

### Honest caveat (the one thing this gate cannot rule out)

v5 **saturates at 100%** vs the hand-built fixtures, so these are not *contested* games —
we are testing "does anticipation help when you already dominate 100–0", not "does it help
in a close game vs a *peer* non-producer agent." Hand-built fixtures cannot reach
producer-tier (the same ceiling that makes the public pool 99%), so a contested off-mirror
instrument was not constructible here. Strictly, this folds "injection helps dispatch
weaker non-producer opponents"; it *weakens but does not formally kill* "injection helps
in contested play vs a strong non-producer peer." Reviving the head would require a
**contested** off-mirror instrument — a non-producer opponent strong enough that v5 wins
~50–70%, e.g. a vendored real top-tier non-producer replay-bot — which is expensive and
previously judged low-EV. Given Cluster 11 (mirror redundancy) **and** this result
(off-mirror inertness on a sensitive, non-saturated margin instrument), the weight of
evidence says fold.

## What survives (reusable infra)

- `agents/external/{archetype_common,half_drainer,swarmer}.py` — non-producer fixtures
  for any future off-mirror experiment.
- `scripts/off_mirror_gate.py` — the off-mirror gate (paired margin instrument, resumable).
- `outputs/arena/offmirror_{half_drainer,swarmer}.csv` — the n=120×4 dose-curve data.
- `agents/v5/main.py` unchanged (`opp_inject_waves=0` default, byte-identical to producer).
