# Leaderboard Climb Plan — 2026-06-09

**Where we are:** rank **1762 / 4143** ("Oshbocker", score **736.7**), agent = ExIt
`v2_exit_a100/ckpt_000020.pt` submitted 2026-06-04. Final submission deadline
**2026-06-23** (kaggle CLI). Only the **latest 2 submissions** are tracked/scored —
our older rule-based subs (apex 756.7, apex3 774.7, hybrid v1 780.5) are inactive.

## The research findings that reframe everything

### 1. The leaderboard meta is far above apex — our anchor was wrong

| LB score | Meaning |
|---|---|
| 686 | median (p50) |
| **736.7** | **us (ExIt iter-20)** — rank 1762 |
| 756–780 | our own rule-based agents (inactive) |
| 931 | p75 |
| 958–1100 | strong *public* heuristic notebooks |
| 1185 | p90 |
| 1224–1287 | best public agents (Tamrazov 1224 hand-tuned; **Producer flow-diff 1287.1**) |
| 1369 | p99 |
| 1723 | #1 (Isaiah @ Tufa Labs) |

Apex ≈ 757 ≈ p55. Every week of ExIt work optimized win-rate against a median-tier
opponent. Even a hypothetical 100%-vs-apex agent could rate ~900 if it loses to the
950–1300 public tier it has never trained against.

### 2. What the 950–1300 public tier actually does (10 notebooks analyzed)

One shared rule-base lineage dominates (pilkwang structured-baseline → Tamrazov/ykhnkf
v4/v6 → forks at 958/1000/1100/1224). Its pillars, none of which apex has:

- **Arrival ledger + exact per-planet ownership timeline** (production + engine-exact
  combat replay to horizon ~110): gives `fall_turn`, `keep_needed` (binary-searched
  min garrison that survives), and `min_ships_to_own_by(target, turn)` (binary-searched
  exact send size including own planned commitments this turn). Replaces ALL
  `garrison + prod*eta + 1` heuristics.
- **Send-size↔speed fixed point** (size→speed→ETA→garrison growth→size, ~4 iters)
  — naive sizing systematically under-sends because bigger fleets fly faster.
- **Synchronized multi-source swarms** (2–3 sources, ETA spread ≤1–2, only when no
  smaller subset suffices, joint outcome re-simulated before commit).
- **Snipe timing**: land at `enemy_eta ± 1` on contested neutrals so the enemy pays
  the garrison cost.
- **Crash exploitation (4P)**: detect two different enemies' fleets arriving at the same
  planet within 2–3 turns → arrive `crash+1` and vulture the survivors.
- **Reaction-time race triage of neutrals**: my fastest legal ETA vs enemy's fastest
  → safe ×1.2 / contested ×0.7; opening avoids rotating neutrals.
- **Reserves & budgets**: attack budget = ships − max(keep_needed, ~18–28% of nearest
  enemy garrison reachable ≤12 turns); frontier planets keep deterrence garrisons.
- **Logistics**: rear planets ship 60–70% of garrison to front staging planets;
  doomed planets evacuate (ships in flight still score); endgame total-war dump.
- **4P meta**: attack the *nearest* opponent (1.25×/0.55× near/far priority — worth
  ~+100 LB by itself per the 1100 fork), elimination drive on the weakest enemy,
  leader-guard variants.
- **Time budgeting**: soft deadline 0.82×actTimeout, heavy phases gated, graceful bail.

### 3. The best public agent (Producer, 1287.1) is conceptually different — and public

Slawek Biel's "Producer" (forked by Tamrazov as `orbit-wars-i-m-better`, self-contained
payload; extracted locally to `/tmp/imbetter_payload/`): a torch-vectorized **exact
counterfactual flow-diff planner**. Per turn: project all planets+fleets 18 turns
(13 in 4P) with engine-exact combat; candidates = top-12 sources × (top-12 attack +
top-4 defend targets); ONE send size per pair = `safe_drain` (max ships the source can
shed and provably still hold itself); score = re-simulate only touched planets and take
`Δnet_me − ΣΔnet_opp`; fire greedily while score > ROI threshold 1.5; leftover ships
regroup up an enemy-pressure gradient. No hand value formula, no fractions, defense and
offense compete in one scorer. **Verified locally 2026-06-09: runs fine under
`kaggle_environments` (torch CPU, ~20–35s/game) and beat apex 4/4, side-alternated,
seeds {101, 202}** (`/tmp/producer_vs_apex.py`).

### 4. Public ML/RL findings replicate ours — and show the working pattern

- Their PPO-from-scratch, PPO+curriculum, single- and multi-teacher SFT all collapsed
  to ~0% vs strong rule-bases (mirrors our PPO dead end).
- The two ML patterns that *work* publicly:
  - **Reject-only shot validator** (konbu17): 24-feature MLP (~5K params) trained on
    dense per-shot supervised labels ("did we own the target within [arrival,
    arrival+10]?"), veto threshold 0.4, self-reinforcements exempt. **+19pp locally,
    gains concentrated vs the strongest opponents, no opponent regressed.** Fail-safe:
    can only drop bad shots from a strong rule-base.
  - **Learned global value for candidate re-ranking** (aidensong): 16-feature GBC,
    (state → P(win)) trained on top-agent replays + self-play, AUC 0.976. Global value
    is very learnable here (consistent with our Phase-2 diagnostic: value grounded
    globally, corr 0.39, but can't rank near-equal siblings).
- Episode data for training: **Meta Kaggle** `Episodes.csv` (5.6GB) /
  `EpisodeAgents.csv` (19GB), refreshed daily; per-episode replay JSON via
  `GetEpisodeReplay`.

### 5. Implications for the in-flight Colab experiments

The ckpt-gate / c_scale / embed A/Bs all optimize win-rate **vs apex** (≈ LB ~757).
Still worth collecting the results (they validate the Gumbel machinery), but shipping
their winner moves us from 736 to at best ~760-territory. The leaderboard lever is the
opponent/teacher pool, not the search hyperparameters.

## Implementation plan (1 week)

**Strategy: re-anchor everything from apex (p55) to the public 1224–1287 tier (p97), by
(a) standing on the best public agents instead of competing with them from below, and
(b) pointing our ML machinery (BC, ExIt, shot-validator) at the new tier.**

### Phase 0 — Stop the bleeding (today, ~1h)
1. ✅ DONE (2026-06-10). Vendored 7 public agents into `agents/external/` (producer
   pkg, tamrazov_1224, distance_1100, shot_validator_hybrid + decoded weights.npz,
   enders_1000, ow_proto, reinforce_958), loaded via
   `agents.external.load_agent(name)` — fresh exec per game (they keep module-level
   ledgers; note: frozen dataclasses require registering the module in `sys.modules`
   before exec). Public Apache-2.0 notebooks; forking is the norm in Kaggle sim comps.
   `agents/external` excluded from ruff/pyright.
2. ✅ DONE (2026-06-10). `scripts/arena.py`: round-robin on the REAL Kaggle env
   (external agents consume full Kaggle obs), side-alternated paired seeds, one
   process per game, incremental CSV with resume (bump --games to extend), win-rate
   matrix output, `exit:<ckpt>:<config>` specs for our checkpoints. This replaces
   "win-rate vs apex" as THE metric. ~15–35s/game, parallel.
3. ✅ DONE (2026-06-10). **Producer bundle submitted** (`outputs/submissions/
   producer_bundle.tar.gz` = main.py + orbit_lite/ at archive root, exact replica of
   the proven i-m-better packaging). Verified pre-submit by extracting the tarball and
   running a real game through Kaggle's file-agent loader (append→exec→pop sequence) —
   status DONE, won vs random. Active slots now: producer bundle (PENDING) + ExIt
   iter-20 (drifted 736.7 → 725.3). Judge the score after ≥1 day of episodes.

**First smoke matrix (2026-06-10, n=2/pair, seeds 20000–20001, side-alternated, 2P
only, 0 errors — `outputs/arena/smoke.csv`; coarse, gate decisions need n≥30):**

| agent | mean | notes |
|---|---|---|
| producer | **100% (18-0)** | swept every opponent — confirmed Phase-1 base + submission candidate |
| ow_proto | 72% | stronger locally than its LB ~1080 suggests |
| **exit embed256 last** | **67%** | beat tamrazov_1224, distance_1100, apex 2-0 each — our RL stack is already mid-public-tier |
| enders_1000 | 56% | |
| exit iter-20 (submitted, LB 736.7) | 44% | mid-pack, consistent with LB |
| tamrazov_1224 | 39% | LB 1224 yet weak here → its rating likely leans on 4P games |
| shot_validator_hybrid / reinforce_958 / apex | 33% | apex confirmed bottom-tier |
| distance_1100 | 22% | its +100 LB came from 4P features — invisible in this 2P-only arena |

**Implication:** the 2P arena under-measures 4P specialists (tamrazov_1224,
distance_1100). Add a 4-player arena mode before trusting the matrix for
submission gating — LB episodes mix both formats.

### Phase 1 — "v5 base": best public skeleton + our orthogonal fixes (days 1–3)
Pick the arena winner as the base (expected: Producer flow-diff). Add what it
measurably lacks (each gated on arena games vs the full pool):
- ~~Comet handling~~ **CLOSED 2026-06-10 — premise was wrong.** Producer does NOT
  ignore comets: `_apply_comet_paths` projects exact comet path positions + expiry
  (`alive_by_step`), and `safe_drain` + the exact flow-diff implicitly evacuate
  doomed comets (saved ships score ≈ their count). Audit (producer vs 1224, seeds
  31000/31001): producer captured 2/1 comets, lost only 1/7 ships to expiry; the
  1224 agent with "exact comet logic" lost 33. Nothing to port.
- ✅ Endgame (2026-06-10, in `agents/v5/`): planning horizon clamped to remaining
  game steps (config-level replace; `garrison_status(max_horizon<build)` is broken
  upstream). Makes the flow-diff exact wrt termination → late capture-refusal +
  total-war drains fall out for free. Gate pending.
- Snipe + crash-exploit timing missions (from the v4 lineage) if the flow-diff scorer
  doesn't already subsume them (check in arena).
- ✅ 4P preset tuning (2026-06-10, in `agents/v5/`): nearest-opponent priority
  (1.25×/0.55× score mult on enemy-owned targets, distance-1100 lineage) added on
  top of Producer's leader-guard bonuses; knobs default-off, on in the 4P preset.
  Gate pending (4P arena).
- ROI threshold / horizon sweep (1.5/18 are hand-tuned for *their* pool, not the
  current LB meta).

**v5 status 2026-06-10:** fork = `agents/v5/` (package renamed `orbit_lite_v5`),
arena spec `v5`, bundle builder `scripts/build_v5_bundle.py` (verified via Kaggle
file loader). Gate runs queued: 2P v5-vs-producer n=60 + 4P co-occurrence n=32,
paired seeds (`outputs/arena/gate_v5*.csv`).

### Phase 2 — ML edge on top (days 3–6, the RL-learning track)
In order of expected value-per-day:
1. **Shot-validator veto** on the v5 base: harvest dense per-shot labels from arena
   self-play vs the strong pool (we already have fast_env batching), train the 24-dim
   MLP locally, bolt on with threshold sweep. Proven +19pp pattern, fail-safe by
   construction, CPU-trainable in hours.
2. **Re-target the ExIt pipeline**: BC teacher and eval/opponent pool switch from apex
   to {producer, 1224, v5}. The Gumbel search machinery (our one validated search win,
   +9.4%) stays; the expert improves because the policy prior and the opponents are
   ~500 LB points stronger. This is the same pipeline we already have — only the data
   distribution changes.
3. (Stretch) **Learned value from real episodes**: pull top-team episodes via Meta
   Kaggle, train the 16-feature global value, use it for candidate re-ranking in the
   flow-diff planner (re-rank near-ties only — respects our sibling-ranking finding).

### Phase 3 — Ship & iterate (continuous)
- Every improvement that beats the incumbent in the arena (n≥60, multi-seed, both 2P
  and 4P) gets submitted same-day; keep the better of the 2 active slots pinned.
- Track LB score deltas per submission in this file.

**Submission log:**
| date | bundle | LB score | note |
|---|---|---|---|
| 2026-06-04 | v2_exit_a100 iter-20 | 736.7 → 725.3 (drifting) | ExIt champion vs apex |
| 2026-06-10 15:23 | producer_bundle | 695.0 (hours old) | ladder starts low; judge ≥1 day |

### Explicit non-goals this week
- No more search-hyperparameter A/Bs vs apex (collect the in-flight Colab results,
  then retire apex as the gate).
- No PPO revival; no neural leaf values inside search (4× confirmed negative).

## Risks
- **Attribution/originality**: Kaggle norms require crediting forked public work in
  writeups; fine for ranking. Our differentiation = comets + 4P tuning + shot
  validator + ExIt-on-top.
- **Public-agent saturation**: many teams run the same public bases; ties in the
  1100–1300 band are expected. The ML edge (Phase 2) is what breaks the tie — almost
  nobody's published ML actually works.
- **Ladder noise**: Tamrazov: "ladder scores are noisy, reruns move up or down" —
  judge submissions after ≥1 day of episodes, not on validation score.
