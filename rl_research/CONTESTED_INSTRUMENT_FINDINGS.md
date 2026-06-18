# Contested non-producer instrument — findings (2026-06-17)

**Goal.** Rec #1 from the v5.3 weakness analysis: build a *contested* off-mirror instrument —
a NON-producer opponent strong enough that v5 wins only ~50–70% — to measure v5's
exploitability against a structurally different style (half-drain / swarm) with win-rate
headroom. The existing hand-built fixtures (`half_drainer`, `swarmer`) lose to v5 **100%**
(saturated → the `OFF_MIRROR_INJECTION_FINDINGS.md` margin instrument read ≈0). A contested
instrument would re-open the opponent-modeling head (Cluster 11) and the half-drain axis
(Clusters 7/9/12) for measurement.

## Verdict — NEGATIVE. No contested non-producer instrument is reachable at acceptable cost.

Three independent constructions were built and measured; **all land sub-1000 tier** (v5 and
even the weakest public agents win ~100%), so none provides the win-rate headroom a contested
instrument needs.

| construction | best result | tier |
|---|---|---|
| **Hand-built strong half-drainer** (`agents/external/contested_drainer.py`) | 0% vs producer/reinforce_958/ow_proto; 8% vs enders/distance/tamrazov | sub-1000 |
| **Cluster BC** (3 half-drainers, 2.9K examples, launch_acc 0.40) | 0% vs producer | sub-1000 |
| **Single-teacher BC of Low-Orbit Losers** (1273; 12K clean examples, launch_acc 0.226) | 0% vs producer; 0% vs reinforce_958; 6% vs the hand-built drainer | sub-958 |

Each *expands* well — the LoL clone reaches **planets@50 ≈ 6–7** (producer-tier opening) and
plays full 180–310-step games — but is overrun in the mid/late game. Not a bug (coherent
play, full games); a fundamental tier ceiling.

## Why (two compounding root causes)

1. **The tempo tax of partial sends.** A half-drainer/swarmer sends capture-minimal /
   fractional fleets. Smaller fleets fly slower (`speed = 1 + (maxSpeed−1)·(ln(ships)/ln1000)^1.5`),
   so a partial-send agent cedes raw expansion + close-out tempo and gets out-produced once the
   full-drain opponent grabs the map (observed every time: even-ish at step ~40, opponent at
   2–3× planets by step ~60). The real top-tier non-producers (Isaiah 1762, LoL 1273) overcome
   this only because their *entire* planner is co-designed around the style. A strong-execution
   layer bolted onto partial sizing is just a degraded full-drainer — the same lesson as the
   half-drain graveyard (Clusters 7/9/12: "a producer-with-a-cap is a degraded producer").

2. **Pointer-BC caps far below its teacher.** Cloning the teacher's target *selection* (the
   AlphaStar pointer decomposition, exact fleet-tracked labels, analytic sizing at exec — the
   design meant to beat the 3%-vs-producer fraction-regressing BC) plateaus at **launch_acc
   ≈ 0.2** for LoL and **does not improve with 7.5× data** (1.6K→12K examples: 0.196→0.226).
   LoL is a high-target-entropy half-drainer (capture_min dominant → many near-equivalent cheap
   targets → top-1 prediction is inherently low even for a perfect policy model), so the clone
   picks *a* reasonable target, just not *the* one — coherent play, but it neither reproduces
   the teacher's exact selection nor its tempo. This corroborates the producer-BC precedent
   (BC clones land far below their teacher) and the rich-BC-selection closure
   (`RICH_BC_SELECTION_FINDINGS.md`).

## What this means for Rec #1

The premise "vendor/reconstruct a strong non-producer instrument" is harder than hoped. A
faithful contested instrument would require either (a) the teacher's **actual code**
(unavailable — replays are state/action logs, not runnable), or (b) a **high-fidelity
full-policy clone** (selection + exact sizing + multi-wave timing + defense), a large build
that the BC precedent says still caps below teacher. **EV is low for the 06-23 deadline.**
Since the instrument was *infrastructure to unlock other measurements* (not a ship), the
pragmatic call is to **redirect to direct-ladder-EV work that does not depend on it** — Rec #2
(4P closing logic, our measured strength: 4P fitted 1290 vs 2P 1186).

## What survives (reusable infrastructure)

- **`scripts/harvest_teacher.py`** — replay-corpus harvest with **pagination**
  (`list_day_files_paged` walks the full ~4600-episode day past the ~200-file single-page cap),
  LB-rating join, per-team teacher ranking by data volume + style. Reusable for any future
  replay-data effort (e.g. mining LoL's selection as a *structural delta*, the proven channel).
- **`scripts/teacher_bc.py`** — replay → fleet-tracked macro labels (reuses
  `scripts/macro_relabel.resolve_launches_for_step`, verified 97% resolve on LoL) → v2 OrbitNet
  pointer-BC. Swap `--teams` to clone any team.
- **`agents/external/bc_teacher.py`** — STANDALONE executor (net selection + capture-minimal
  analytic sizing + `OW_BC_FIRE_MARGIN` aggression knob); does NOT route through v5's planner
  (which would full-drain and collapse to the mirror). Falls back to the heuristic drainer when
  no checkpoint is present.
- **`agents/external/contested_drainer.py`** — the strong hand-built half-drain planner
  (consolidation + defense + production-first targeting), tunable via `Config`.
- Cached corpus `outputs/replay_pulse/cache/2026-06-12/` (~600 episodes) + labeled datasets
  `outputs/teacher_bc/dataset_lol_big.npz` + checkpoints `outputs/checkpoints/bc_teacher/`.

## Honest caveats (what this does NOT rule out)

- It does **not** prove a non-producer style *cannot* be top-tier — the real LoL/Isaiah are.
  It proves we cannot cheaply *reconstruct* one as a measurement fixture.
- A **full-policy** BC (cloning sizing + multi-wave timing, not just selection) was not tried;
  the partial result (good expansion, lost tempo) hints the missing piece is the execution
  layer, but the BC-caps-below-teacher precedent makes its EV low for the deadline.
- The teacher pool was 06-12 (top ~1346 in sample; the 1500–1800 tier was not captured —
  pagination reaches it but those agents are rarer). A stronger teacher might clone to a higher
  (still likely sub-teacher) tier, but the tempo tax + launch_acc ceiling apply regardless.
