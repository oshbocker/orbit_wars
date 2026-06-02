# Phase 5 (+6) Plan — Neural-Value ExIt Search & Replay Data

*Next-session execution plan. Phases 0–4 of the v4 ceiling agent are DONE & tested
(see `rl_research/V3_STALL_PLAYBOOK.md` for the why). This covers the two deferred
"ceiling" phases: **Tier 3.2** (ExIt search scored by OrbitNet's value head) and
**Tier 3.3** (BC/value data from top-LB replays).*

## Prerequisite (do NOT start Phase 5 before this)
Phase 5 only makes sense once a **trained v4 value head** exists. Neural-value
search on an untrained critic = noise ("garbage value → garbage search targets").
So: train `configs/v4_ceiling.yaml` first, get a checkpoint with a non-trivial
apex win-rate, THEN build Phase 5 and test the search against that checkpoint.

## Config flags already in place (added in Phase 0, default-off)
- `exit.neural_value_leaves: bool` — score search leaves with the value head.
- `exit.use_batched_env: bool` — run candidate rollouts on fast_env.
- `exit.two_player_search: bool` — in-sim opponent during search.
- `imitation.bc_replay_path: str` — extra demos from top-LB replays (Phase 6).
`configs/v4_ceiling.yaml` already sets the exit block (neural_value_leaves: true,
two_player_search: true) — but `v2.exit_train` ignores these until coded.

## Files to read first (next session)
1. `src/simulator.py` (198 lines) — `SimState`, `sim_step`, `evaluate_state`,
   `build_sim_state`, `add_fleet_event`, `travel_time`. **KEY FACT:** `SimState`
   is **positionless** — `{planet_owner, planet_ships, planet_prod, fleet_events,
   current_step, planet_ids}`. No x/y. This is the central obstacle.
2. `v2/search.py` (125 lines) — `search_improve_planet` does the per-source
   candidate sweep; leaves scored by `evaluate_state(sim, player)` (handcrafted).
3. `v2/exit_train.py` (375 lines) — `collect_games` → `search_improve` →
   `train_epoch` (distill target+frac+value). Search call site: `_search_record`
   (~line 156) calls `search_improve_planet`.

## The core obstacle & fix
OrbitNet's value head needs geometric features (positions, orbit, KNN, pair
travel-times). `SimState` has none. So "score leaf with neural value" requires
reconstructing an approximate `GameState` at each leaf:
- Take planet **positions/radius/orbit** from the **root** `GameState` (static
  planets don't move; orbiting planets drift only slightly over `search_depth`
  steps — acceptable for a leaf estimate; optionally advance orbit by depth).
- Take **owner/ships/production** from the leaf `SimState`.
- Fleets: approximate as empty (or reconstruct from `sim.fleet_events` still
  in flight) — losing in-flight detail at the leaf is acceptable.

## Implementation steps
**A. `v2/search.py`**
1. `_simstate_to_gamestate(sim, root_state) -> GameState`: build `PlanetState`s
   using root positions + leaf owner/ships/prod; keep only planets in
   `sim.planet_owner`; fleets=[].
2. `make_neural_value_fn(model, root_state, env_cfg, device, value_norm) -> fn(sim, player)->float`:
   reconstruct GameState → `encode_features` → `model(...).value` →
   **denormalize** (symexp if value_symlog, or `value_norm.denormalize`).
   Batch leaves where possible: collect all candidate leaf SimStates for a
   source planet, encode+stack, ONE forward, read values — big speedup vs
   per-leaf forward.
3. Add `value_fn=None` param to `search_improve_planet`; default to
   `evaluate_state`; replace the two `evaluate_state(...)` calls with `value_fn`.

**B. `v2/exit_train.py`**
4. In `search_improve` / `_search_record`, when `cfg.exit.neural_value_leaves`,
   build the neural `value_fn` (pass model/device + a ValueNorm if used) and
   thread it into `search_improve_planet`. Keep heuristic default otherwise.
5. (`two_player_search`) Currently the simulator advances with no opponent. For
   fidelity, step candidate rollouts on `v2/fast_env.py` (2-player) with an
   in-sim opponent acting (apex or frozen-self). This is a bigger change —
   stage it after the simpler positionless-leaf version works.

**C. Smoke tests**
6. `_simstate_to_gamestate` round-trips a known state (owners/ships preserved).
7. `make_neural_value_fn` returns finite floats; higher for winning positions
   (hand-build a dominant-vs-losing SimState, assert value ordering).
8. Run `v2.exit_train` for 1 iteration on `configs/v4_ceiling.yaml` (exit block)
   with a trained v4 checkpoint resumed → search improves, distill loss drops.

## Phase 6 (Tier 3.3) — replay data
- Pull `penguin069/orbit-wars-local-arena` (kaggle datasets download); it has
  `replay_data.js` (~3MB of replays) + an arena harness.
- Parse replays into (obs, expert_move) for BC and (state, winner) for value
  pretrain. Map expert moves to v4 target/frac slots via the existing angular
  matching in `v2/imitation.py` (`_map_expert_moves_to_v2`).
- Feed via `imitation.bc_replay_path`; blend into `v2_bc_pretrain` /
  `collect_v2_demonstrations`. Stronger teacher than apex alone.

## Quick orientation for next session
The v4 architecture/training code is fully built and tested through Phase 4.
Search (`v2/search.py`) + ExIt loop (`v2/exit_train.py`) are the only files to
modify for Phase 5. Everything is flag-gated; v2/v3/v4-PPO are unaffected.
</content>
