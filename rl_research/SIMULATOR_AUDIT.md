# Simulator Fidelity Audit — `src/simulator.py` vs the Kaggle engine

**Date:** 2026-05-31
**Ground truth:** `.venv/.../kaggle_environments/envs/orbit_wars/orbit_wars.py` (`interpreter()`)
**Audited:** `src/simulator.py` (`SimState`, `sim_step`, `add_fleet_event`, `evaluate_state`,
`build_sim_state`) + helpers in `src/features.py` (`fleet_speed`, `fleet_hits_planet`,
`passes_through_sun`).
**Purpose:** decide what must be fixed before the simulator can serve as a faithful,
fast self-play training environment (Priority 0 in `IMPROVEMENT_RESEARCH.md`).

---

## Verdict

The current simulator is a **single-player, fleet-as-scheduled-event approximation**. It is
adequate for *coarse* action ranking (its ExIt use) but is **not faithful enough to train a
competitive agent in** without changes. The gaps fall into three tiers:

- **Tier A (correctness-breaking — must fix):** no opponent, event-based fleets (no
  continuous collision / no sun / edge death), combat garrison double-count, turn-order
  errors, capture formula off-by-semantics.
- **Tier B (fidelity — should fix):** orbit step indexing, no planet sweep, no production
  on the arrival turn handled correctly, comets ignored.
- **Tier C (acceptable approximations):** terminal scoring, neutral handling.

The engine's true turn order (per `interpreter()`):
**0 launch → 1 production (all owned planets +prod) → 2 fleet movement + continuous
collision (planet → else bounds → else sun) → 3 planet rotation + comet movement + sweep →
4 combat resolution → termination/scoring.**

---

## Tier A — correctness-breaking gaps

### A1. No opponent in the simulation *(the big one)*
**Engine:** every agent's moves are processed each step (`process_moves` for all players),
so enemies launch, capture, and defend.
**Sim:** `sim_step` advances only production + *pre-scheduled* fleet arrivals + combat. There
is **no opponent policy** — enemy planets never launch, never reinforce, never attack.
**Impact:** a policy trained here learns against a passive world: over-extending and leaving
home undefended is never punished, and aggression has no downside. This is the root cause of
the ExIt fraction head collapsing to uniform (no fraction is ever "wrong"). **Any training
env built on this must inject an opponent that acts inside the sim** (apex, or a frozen
self), processed in the same launch phase as the learner.

### A2. Fleets are scheduled arrival events, not moving bodies
**Engine:** fleets move `speed` units/turn along a ray; **continuous collision** each turn
checks (a) any planet along the old→new segment (`point_to_segment_distance < radius`),
(b) out-of-bounds death, (c) sun-crossing death. A fleet can hit a *different* planet than
intended, die on the board edge, or be swept by a moving planet.
**Sim:** `build_sim_state` ray-casts each existing fleet once to its *intended* planet via
`fleet_hits_planet` and stores a single `(arrival_step, target, owner, ships)` event;
`add_fleet_event` does the same for new launches. After that, fleets are invisible —
**no per-turn movement, no mid-flight interception, no edge/sun death, no re-routing.**
**Impact:** (i) fleets that the engine would kill (sun/edge/wrong-planet) are counted as
clean arrivals; (ii) a fleet's *position* never exists, so an opponent (A1) can't intercept
it and moving planets can't sweep it (B2); (iii) `fleet_hits_planet` uses `hit_r =
radius + 0.5` whereas the engine uses bare `radius` for the segment test — a small radius
mismatch. **A faithful env must simulate fleets as moving entities with the engine's
continuous-collision order (planet → bounds → sun).**

### A3. Combat double-counts the garrison
**Engine (step 4):** garrison is **not** added to the attacker tally. It sums *arriving
fleet* ships per player; `survivor = top − second`; then applies survivor vs the planet's
existing garrison:
```
if survivor_owner == planet.owner:  planet.ships += survivor
else:                               planet.ships -= survivor
                                    if planet.ships < 0:  owner = survivor_owner; ships = |ships|
```
**Sim (`sim_step`):** it **injects the garrison into `attackers[defender]` before** taking
top-two:
```
attackers[defender] += garrison
top, second = top two
survivors = top - second
... assign owner/ships from survivors
```
**Impact:** different math. Example — planet owned by D with 10 garrison, attacker A sends 8:
- *Engine:* arrivals {A:8}; single player ⇒ survivor A:8; 8 vs garrison 10 ⇒ 10−8=2, **D
  keeps planet with 2**.
- *Sim:* attackers {A:8, D:10}; top D:10, second A:8; survivor D:2 ⇒ **D keeps with 2**.
  (Matches here.)
Now A sends 12 vs D's 10:
- *Engine:* arrivals {A:12}; survivor A:12; 12 vs 10 ⇒ 10−12=−2 ⇒ **A captures with 2**.
- *Sim:* {A:12, D:10}; survivor A:2 ⇒ **A captures with 2**. (Matches.)
Two attackers A:12, B:5 on neutral (garrison 6):
- *Engine:* arrivals {A:12,B:5}; survivor A:7; neutral owner≠A ⇒ 6−7=−1 ⇒ **A captures
  with 1**.
- *Sim:* {A:12,B:5,neutral:6}; top A:12, second neutral:6; survivor A:6; 6 vs 6 (neutral)…
  ⇒ **A captures with 6**. **MISMATCH** (engine gives 1, sim gives 6).
**Root cause:** the engine resolves *attackers among themselves first*, then the survivor
fights the garrison; the sim lumps the garrison into the attacker pool, so a large neutral/
defender garrison wrongly suppresses the second *attacker*. **Must rewrite combat to the
engine's two-stage form.**

### A4. Production timing / fleet-launch ordering
**Engine order:** launch (deduct ships from source) → **production** → movement → combat.
So a planet that launches still receives production that same turn, and combat ships arrive
*after* production.
**Sim (`sim_step`):** production first, then arrivals/combat — but launches are applied at
*scheduling time* (`add_fleet_event` deducts immediately, outside the step loop). For
multi-step lookahead this means launch deduction and the production schedule can desync from
the engine's within-turn order. **Impact:** small per-turn ship-count drift that compounds
over a long rollout. **Must apply launches inside the step, in engine order.**

### A5. Capture/`survivor` semantics vs neutral
Engine treats neutral (`owner == -1`) garrison as *just a garrison*, never as a "player" in
the arrivals tally. Sim adds `attackers[-1] += garrison`, making neutral a competitor in the
top-two — same bug as A3, called out separately because it also affects the *single-attacker
on neutral* case. Confirm fix removes neutral from the attacker pool.

---

## Tier B — fidelity gaps (should fix for a training env)

### B1. Orbit step indexing
**Engine:** `current_angle = initial_angle + angular_velocity * step`, using the *post-
increment* step, and rotation happens **after** fleet movement, **before** combat. Sim does
not rotate planets at all during `sim_step` (planets are static dicts), so orbiting-planet
positions are frozen at the build-time snapshot. **Impact:** multi-turn lookahead against
orbiting planets aims at stale positions; `intercept_pos` in features compensates at launch
time but the sim's internal projection doesn't. A faithful env must rotate orbiting planets
each step.

### B2. No planet sweep
**Engine (step 3):** a moving planet (orbiting or comet) that sweeps over a fleet's position
captures it into combat (`sweep_fleets`). Sim has no fleet positions (A2) so cannot model
this. Lower-frequency effect but real (inner-system fleets get swept).

### B3. Comets entirely ignored
**Engine:** comets spawn at steps 50/150/250/350/450, move along precomputed elliptical
paths, produce 1 ship, can be captured, and expire. Sim's `build_sim_state` treats them as
ordinary static planets (whatever was in the snapshot) and never spawns/moves/expires them.
**Impact:** for short-horizon lookahead near a spawn step this is wrong; for a full-episode
training env, comets are a real source of production and must be modeled (we already have
the comet path data in the observation).

### B4. `evaluate_state` is a heuristic, not the game score
`evaluate_state` = ship-advantage + `prod_weight·prod-advantage` with a step-dependent
weight (≤15). The **engine's terminal score** is simply `sum(ships on owned planets) +
sum(ships in owned fleets)`, winner = max score. For a *training* env we want the true sparse
±1 terminal (plus optional PBRS), not this heuristic. Keep the heuristic only as an optional
leaf-value for search; don't let it define the reward.

---

## Tier C — acceptable approximations

- **Termination:** engine ends at `step >= episodeSteps - 2` or `≤1 player alive`. Easy to
  replicate exactly; currently sim has `MAX_STEPS=500` only.
- **`fleet_speed`:** **matches the engine exactly** (`1 + (max-1)·(ln(ships)/ln(1000))^1.5`,
  capped at max), including the `ships<=1 → 1.0` guard. ✓ Good.
- **`passes_through_sun`:** uses `SUN_SAFE_RADIUS = SUN_RADIUS+2` (a safety margin) vs the
  engine's bare `SUN_RADIUS` for fleet death. Fine as a *launch filter* (conservative), but
  the *env* must use bare `SUN_RADIUS` for the death check to match the engine. ✓ note.
- **Constants** (BOARD_SIZE=100, SUN_RADIUS=10, CENTER=50, max speed 6): all match. ✓

---

## What to build (Priority 0 plan, informed by this audit)

A new **`v2/fast_env.py`** that re-implements the engine's exact turn loop on lightweight
arrays, with these non-negotiables from the audit:

1. **Opponent acts in-sim** (A1): process learner + opponent launches in one phase.
2. **Fleets are moving bodies** (A2): per-turn ray step + continuous collision in engine
   order (planet → bounds → sun); store positions.
3. **Engine-exact combat** (A3/A5): arrivals-only top-two among players, *then* survivor vs
   garrison; neutral is garrison-only.
4. **Engine turn order** (A4): launch → production → movement/collision → rotation+comet+
   sweep → combat → terminate.
5. **Orbit rotation each step** (B1) and **sweep** (B2).
6. **True terminal score** (B4) for reward; heuristic only as optional search leaf value.
7. Comets (B3) can be a **fast-follow** (model them once 1–5 land in fidelity tests).

**Fidelity test (the gate):** run identical (agent, opponent, seed) episodes through the
Kaggle engine and `fast_env`; assert per-step planet ownership/ship counts and fleet
counts match (allow tiny float tolerance), and that terminal scores/winners agree across a
batch of seeds. Only after this passes do we trust self-play trained in `fast_env`.

**Reuse:** `fleet_speed`, `passes_through_sun`, the collision math in `fleet_hits_planet`,
and the orbit formula in `intercept_pos`/`planet_pos_at` are all correct and portable. Once
the scalar `fast_env` matches the engine, **batch it** (fixed `[N,P]`/`[N,F]` arrays, masked
ops) for the 10–100× throughput that unlocks self-play.

---

## BUILD STATUS (2026-05-31) — scalar `v2/fast_env.py` complete & fidelity-verified

`v2/fast_env.py` (`FastOrbitWars`) implements the full engine turn loop with all Tier-A/B
fixes: opponent acts in-sim (A1), fleets are moving bodies with continuous collision in
engine order (A2), engine-exact two-stage combat (A3/A5), correct turn order (A4), orbit
rotation each step (B1), planet/comet sweep (B2), comets spawn/move/expire (B3), and true
terminal scoring (B4). It reuses the engine's own `generate_planets` / `generate_comet_paths`
/ `point_to_segment_distance`, so map-gen and geometry are fidelity-free by construction.

**Fidelity gate (`scripts/test_fast_env_fidelity.py`): PASSED.** 16/16 episodes match the
Kaggle engine **step-for-step** (every planet owner/position/ships + fleet counts), across
200–300 step episodes that include comet spawns at 50/150/250. Two divergences found during
bring-up were both *test-harness artifacts*, not simulator bugs: (1) the engine applies one
production turn during its init no-op (fixed by aligning both to step=1); (2) the engine only
populates `step` on player 0's obs (fixed by using a step-independent scripted policy). The
simulator logic matched on the first real comparison once those were controlled for.

**Measured scalar speedup: ~1.5×** (123 vs 81 steps/s, single process). This is modest — the
raw Kaggle `interpreter()` is faster than assumed; the training slowness came from the
*harness wrapper + feature encoding + opponent inference*, not the interpreter. So the value
of fast_env is **not** the scalar port. It is:
  1. **Zero harness/obs-dict overhead** in the training hot loop (act directly on arrays).
  2. **Batching** — no `kaggle_environments` dependency in the step loop, so it can be
     vectorized to step N games at once (the real 10–100× / JAX 1000× path).

**Next:** (a) a thin RL adapter so self-play trains against an in-sim opponent (apex or
frozen self) directly on fast_env; (b) the batched/vectorized version for throughput; (c)
keep the fidelity gate as a regression test for both.
