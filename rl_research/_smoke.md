# Clone-Residual Divergence Mine — Findings

_Counterfactual action-diff of top-tier producer-family agents vs bare `producer` and our `v5`, on the IDENTICAL observation each agent saw. Discovery only — gate at n>=100 mirror A/B before shipping._

- Target seats analyzed: **4**; classified **posture-clone** (median send-fraction >= 0.9 AND launches/source <= 1.4): **4**; clone decision-turns: **550**.

- **Headline:** the top tier is producer-family in POSTURE (full-drain, ~1 wave/source) yet its (source,target) SELECTION diverges sharply from bare producer — the residual is in WHAT to attack and FROM WHERE, not how much to send. Tables below quantify where/how.


## Seat roster

`Jaccard` = mean per-turn (src,tgt) overlap with producer's counterfactual (LOW even for full-drain clones = the residual). `wps` = launches per source.

| team | rank | rtg | fmt | turns | med frac | wps | Jaccard vs prod | posture-clone? |
|---|---|---|---|---|---|---|---|---|
| Jake Will | 1 | 1714 | 2P | 173 | 1.00 | 1.00 | 0.08 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 173 | 1.00 | 1.00 | 0.12 | ✅ |
| Hober Malloc | 7 | 1543 | 2P | 102 | 1.00 | 1.00 | 0.11 | ✅ |
| Vadasz & Ascalon | 9 | 1495 | 2P | 102 | 1.00 | 1.00 | 0.06 | ✅ |

## Selection divergence (clone vs baseline, same obs)

`pair_prec` = of the clone's launches, fraction producer also makes (low = clone makes launches producer wouldn't = different/extra targets). `pair_rec` = of producer's launches, fraction the clone also makes (low = clone SKIPS launches producer makes). `src_*` = same at source-planet granularity. `tgt_match|src` = on shared source planets, fraction with the SAME target. 1.0 = identical to producer; lower = more divergent.

| vs | metric | state class | n | mean | sd |
|---|---|---|---|---|---|
| prod | pair_prec | all=all | 304 | 0.227 | 0.356 |
| prod | pair_prec | contested=uncontested | 304 | 0.227 | 0.356 |
| prod | pair_prec | fmt=2P | 304 | 0.227 | 0.356 |
| prod | pair_prec | fmt=2P|contested=uncontested | 304 | 0.227 | 0.356 |
| prod | pair_prec | fmt=2P|phase=mid | 185 | 0.227 | 0.339 |
| prod | pair_prec | fmt=2P|phase=open | 119 | 0.227 | 0.381 |
| prod | pair_prec | fmt=2P|standing=ahead | 188 | 0.236 | 0.356 |
| prod | pair_prec | fmt=2P|standing=behind | 116 | 0.212 | 0.355 |
| prod | pair_prec | phase=mid | 185 | 0.227 | 0.339 |
| prod | pair_prec | phase=open | 119 | 0.227 | 0.381 |
| prod | pair_prec | standing=ahead | 188 | 0.236 | 0.356 |
| prod | pair_prec | standing=behind | 116 | 0.212 | 0.355 |
| prod | pair_rec | all=all | 407 | 0.134 | 0.273 |
| prod | pair_rec | contested=uncontested | 407 | 0.134 | 0.273 |
| prod | pair_rec | fmt=2P | 407 | 0.134 | 0.273 |
| prod | pair_rec | fmt=2P|contested=uncontested | 407 | 0.134 | 0.273 |
| prod | pair_rec | fmt=2P|phase=mid | 282 | 0.106 | 0.220 |
| prod | pair_rec | fmt=2P|phase=open | 125 | 0.197 | 0.357 |
| prod | pair_rec | fmt=2P|standing=ahead | 209 | 0.162 | 0.292 |
| prod | pair_rec | fmt=2P|standing=behind | 198 | 0.105 | 0.248 |
| prod | pair_rec | phase=mid | 282 | 0.106 | 0.220 |
| prod | pair_rec | phase=open | 125 | 0.197 | 0.357 |
| prod | pair_rec | standing=ahead | 209 | 0.162 | 0.292 |
| prod | pair_rec | standing=behind | 198 | 0.105 | 0.248 |
| prod | src_prec | all=all | 304 | 0.467 | 0.418 |
| prod | src_prec | contested=uncontested | 304 | 0.467 | 0.418 |
| prod | src_prec | fmt=2P | 304 | 0.467 | 0.418 |
| prod | src_prec | fmt=2P|contested=uncontested | 304 | 0.467 | 0.418 |
| prod | src_prec | fmt=2P|phase=mid | 185 | 0.495 | 0.411 |
| prod | src_prec | fmt=2P|phase=open | 119 | 0.422 | 0.426 |
| prod | src_prec | fmt=2P|standing=ahead | 188 | 0.469 | 0.421 |
| prod | src_prec | fmt=2P|standing=behind | 116 | 0.463 | 0.415 |
| prod | src_prec | phase=mid | 185 | 0.495 | 0.411 |
| prod | src_prec | phase=open | 119 | 0.422 | 0.426 |
| prod | src_prec | standing=ahead | 188 | 0.469 | 0.421 |
| prod | src_prec | standing=behind | 116 | 0.463 | 0.415 |
| prod | src_rec | all=all | 407 | 0.260 | 0.339 |
| prod | src_rec | contested=uncontested | 407 | 0.260 | 0.339 |
| prod | src_rec | fmt=2P | 407 | 0.260 | 0.339 |
| prod | src_rec | fmt=2P|contested=uncontested | 407 | 0.260 | 0.339 |
| prod | src_rec | fmt=2P|phase=mid | 282 | 0.216 | 0.293 |
| prod | src_rec | fmt=2P|phase=open | 125 | 0.360 | 0.408 |
| prod | src_rec | fmt=2P|standing=ahead | 209 | 0.307 | 0.351 |
| prod | src_rec | fmt=2P|standing=behind | 198 | 0.211 | 0.319 |
| prod | src_rec | phase=mid | 282 | 0.216 | 0.293 |
| prod | src_rec | phase=open | 125 | 0.360 | 0.408 |
| prod | src_rec | standing=ahead | 209 | 0.307 | 0.351 |
| prod | src_rec | standing=behind | 198 | 0.211 | 0.319 |
| prod | tgt_match|src | all=all | 189 | 0.485 | 0.466 |
| prod | tgt_match|src | contested=uncontested | 189 | 0.485 | 0.466 |
| prod | tgt_match|src | fmt=2P | 189 | 0.485 | 0.466 |
| prod | tgt_match|src | fmt=2P|contested=uncontested | 189 | 0.485 | 0.466 |
| prod | tgt_match|src | fmt=2P|phase=mid | 124 | 0.479 | 0.460 |
| prod | tgt_match|src | fmt=2P|phase=open | 65 | 0.495 | 0.477 |
| prod | tgt_match|src | fmt=2P|standing=ahead | 117 | 0.499 | 0.454 |
| prod | tgt_match|src | fmt=2P|standing=behind | 72 | 0.461 | 0.485 |
| prod | tgt_match|src | phase=mid | 124 | 0.479 | 0.460 |
| prod | tgt_match|src | phase=open | 65 | 0.495 | 0.477 |
| prod | tgt_match|src | standing=ahead | 117 | 0.499 | 0.454 |
| prod | tgt_match|src | standing=behind | 72 | 0.461 | 0.485 |
| v5 | pair_prec | all=all | 304 | 0.226 | 0.357 |
| v5 | pair_prec | contested=uncontested | 304 | 0.226 | 0.357 |
| v5 | pair_prec | fmt=2P | 304 | 0.226 | 0.357 |
| v5 | pair_prec | fmt=2P|contested=uncontested | 304 | 0.226 | 0.357 |
| v5 | pair_prec | fmt=2P|phase=mid | 185 | 0.233 | 0.344 |
| v5 | pair_prec | fmt=2P|phase=open | 119 | 0.216 | 0.376 |
| v5 | pair_prec | fmt=2P|standing=ahead | 188 | 0.238 | 0.362 |
| v5 | pair_prec | fmt=2P|standing=behind | 116 | 0.208 | 0.349 |
| v5 | pair_prec | phase=mid | 185 | 0.233 | 0.344 |
| v5 | pair_prec | phase=open | 119 | 0.216 | 0.376 |
| v5 | pair_prec | standing=ahead | 188 | 0.238 | 0.362 |
| v5 | pair_prec | standing=behind | 116 | 0.208 | 0.349 |
| v5 | pair_rec | all=all | 394 | 0.153 | 0.299 |
| v5 | pair_rec | contested=uncontested | 394 | 0.153 | 0.299 |
| v5 | pair_rec | fmt=2P | 394 | 0.153 | 0.299 |
| v5 | pair_rec | fmt=2P|contested=uncontested | 394 | 0.153 | 0.299 |
| v5 | pair_rec | fmt=2P|phase=mid | 279 | 0.122 | 0.247 |
| v5 | pair_rec | fmt=2P|phase=open | 115 | 0.227 | 0.389 |
| v5 | pair_rec | fmt=2P|standing=ahead | 205 | 0.185 | 0.319 |
| v5 | pair_rec | fmt=2P|standing=behind | 189 | 0.118 | 0.271 |
| v5 | pair_rec | phase=mid | 279 | 0.122 | 0.247 |
| v5 | pair_rec | phase=open | 115 | 0.227 | 0.389 |
| v5 | pair_rec | standing=ahead | 205 | 0.185 | 0.319 |
| v5 | pair_rec | standing=behind | 189 | 0.118 | 0.271 |
| v5 | src_prec | all=all | 304 | 0.416 | 0.414 |
| v5 | src_prec | contested=uncontested | 304 | 0.416 | 0.414 |
| v5 | src_prec | fmt=2P | 304 | 0.416 | 0.414 |
| v5 | src_prec | fmt=2P|contested=uncontested | 304 | 0.416 | 0.414 |
| v5 | src_prec | fmt=2P|phase=mid | 185 | 0.437 | 0.407 |
| v5 | src_prec | fmt=2P|phase=open | 119 | 0.384 | 0.423 |
| v5 | src_prec | fmt=2P|standing=ahead | 188 | 0.424 | 0.418 |
| v5 | src_prec | fmt=2P|standing=behind | 116 | 0.404 | 0.406 |
| v5 | src_prec | phase=mid | 185 | 0.437 | 0.407 |
| v5 | src_prec | phase=open | 119 | 0.384 | 0.423 |
| v5 | src_prec | standing=ahead | 188 | 0.424 | 0.418 |
| v5 | src_prec | standing=behind | 116 | 0.404 | 0.406 |
| v5 | src_rec | all=all | 394 | 0.272 | 0.359 |
| v5 | src_rec | contested=uncontested | 394 | 0.272 | 0.359 |
| v5 | src_rec | fmt=2P | 394 | 0.272 | 0.359 |
| v5 | src_rec | fmt=2P|contested=uncontested | 394 | 0.272 | 0.359 |
| v5 | src_rec | fmt=2P|phase=mid | 279 | 0.222 | 0.314 |
| v5 | src_rec | fmt=2P|phase=open | 115 | 0.394 | 0.425 |
| v5 | src_rec | fmt=2P|standing=ahead | 205 | 0.327 | 0.375 |
| v5 | src_rec | fmt=2P|standing=behind | 189 | 0.213 | 0.330 |
| v5 | src_rec | phase=mid | 279 | 0.222 | 0.314 |
| v5 | src_rec | phase=open | 115 | 0.394 | 0.425 |
| v5 | src_rec | standing=ahead | 205 | 0.327 | 0.375 |
| v5 | src_rec | standing=behind | 189 | 0.213 | 0.330 |
| v5 | tgt_match|src | all=all | 174 | 0.526 | 0.467 |
| v5 | tgt_match|src | contested=uncontested | 174 | 0.526 | 0.467 |
| v5 | tgt_match|src | fmt=2P | 174 | 0.526 | 0.467 |
| v5 | tgt_match|src | fmt=2P|contested=uncontested | 174 | 0.526 | 0.467 |
| v5 | tgt_match|src | fmt=2P|phase=mid | 113 | 0.529 | 0.456 |
| v5 | tgt_match|src | fmt=2P|phase=open | 61 | 0.519 | 0.488 |
| v5 | tgt_match|src | fmt=2P|standing=ahead | 108 | 0.535 | 0.453 |
| v5 | tgt_match|src | fmt=2P|standing=behind | 66 | 0.510 | 0.489 |
| v5 | tgt_match|src | phase=mid | 113 | 0.529 | 0.456 |
| v5 | tgt_match|src | phase=open | 61 | 0.519 | 0.488 |
| v5 | tgt_match|src | standing=ahead | 108 | 0.535 | 0.453 |
| v5 | tgt_match|src | standing=behind | 66 | 0.510 | 0.489 |

## Ranked behavioural divergences (clone − baseline)

Positive `Δ` = clone does MORE of the axis than the baseline on the SAME obs. `base` = baseline's own mean (for scale); `pct` = Δ as % of base; `t` = t-stat (|t|>=3 kept); `eff` = |Δ|/sd; `cons` = sign-agreement; `nz` = fraction of turns exercising the axis. Ranked by |t|·eff.

| # | vs | axis | state class | n | Δ | base | pct | t | eff | cons | nz |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | prod | ships | phase=mid | 310 | -109.974 | 160.76 | -68% | -12.4 | 0.70 | 0.73 | 0.91 |
| 2 | prod | ships | fmt=2P|phase=mid | 310 | -109.974 | 160.76 | -68% | -12.4 | 0.70 | 0.73 | 0.91 |
| 3 | prod | waves | phase=mid | 310 | -1.377 | 2.52 | -55% | -12.3 | 0.70 | 0.66 | 0.78 |
| 4 | prod | waves | fmt=2P|phase=mid | 310 | -1.377 | 2.52 | -55% | -12.3 | 0.70 | 0.66 | 0.78 |
| 5 | prod | sources | phase=mid | 310 | -1.365 | 2.51 | -54% | -12.2 | 0.69 | 0.66 | 0.79 |
| 6 | prod | sources | fmt=2P|phase=mid | 310 | -1.365 | 2.51 | -54% | -12.2 | 0.69 | 0.66 | 0.79 |
| 7 | prod | ships | all=all | 550 | -68.156 | 109.29 | -62% | -12.1 | 0.51 | 0.57 | 0.77 |
| 8 | prod | ships | contested=uncontested | 550 | -68.156 | 109.29 | -62% | -12.1 | 0.51 | 0.57 | 0.77 |
| 9 | prod | ships | fmt=2P | 550 | -68.156 | 109.29 | -62% | -12.1 | 0.51 | 0.57 | 0.77 |
| 10 | prod | ships | fmt=2P|contested=uncontested | 550 | -68.156 | 109.29 | -62% | -12.1 | 0.51 | 0.57 | 0.77 |
| 11 | v5 | ships | phase=mid | 310 | -83.035 | 133.82 | -62% | -10.6 | 0.60 | 0.66 | 0.91 |
| 12 | v5 | ships | fmt=2P|phase=mid | 310 | -83.035 | 133.82 | -62% | -10.6 | 0.60 | 0.66 | 0.91 |
| 13 | prod | active | phase=mid | 310 | -0.313 | 0.91 | -34% | -10.5 | 0.60 | 0.34 | 0.37 |
| 14 | prod | active | fmt=2P|phase=mid | 310 | -0.313 | 0.91 | -34% | -10.5 | 0.60 | 0.34 | 0.37 |
| 15 | prod | waves | all=all | 550 | -0.873 | 1.85 | -47% | -11.7 | 0.50 | 0.51 | 0.66 |
| 16 | prod | waves | contested=uncontested | 550 | -0.873 | 1.85 | -47% | -11.7 | 0.50 | 0.51 | 0.66 |
| 17 | prod | waves | fmt=2P | 550 | -0.873 | 1.85 | -47% | -11.7 | 0.50 | 0.51 | 0.66 |
| 18 | prod | waves | fmt=2P|contested=uncontested | 550 | -0.873 | 1.85 | -47% | -11.7 | 0.50 | 0.51 | 0.66 |
| 19 | prod | sources | all=all | 550 | -0.865 | 1.84 | -47% | -11.6 | 0.49 | 0.51 | 0.67 |
| 20 | prod | sources | contested=uncontested | 550 | -0.865 | 1.84 | -47% | -11.6 | 0.49 | 0.51 | 0.67 |
| 21 | prod | sources | fmt=2P | 550 | -0.865 | 1.84 | -47% | -11.6 | 0.49 | 0.51 | 0.67 |
| 22 | prod | sources | fmt=2P|contested=uncontested | 550 | -0.865 | 1.84 | -47% | -11.6 | 0.49 | 0.51 | 0.67 |
| 23 | prod | waves | standing=behind | 259 | -1.046 | 1.80 | -58% | -9.8 | 0.61 | 0.56 | 0.68 |
| 24 | prod | waves | fmt=2P|standing=behind | 259 | -1.046 | 1.80 | -58% | -9.8 | 0.61 | 0.56 | 0.68 |
| 25 | prod | sources | standing=behind | 259 | -1.046 | 1.80 | -58% | -9.8 | 0.61 | 0.56 | 0.68 |
| 26 | prod | sources | fmt=2P|standing=behind | 259 | -1.046 | 1.80 | -58% | -9.8 | 0.61 | 0.56 | 0.68 |
| 27 | v5 | active | phase=mid | 310 | -0.303 | 0.90 | -34% | -9.9 | 0.56 | 0.34 | 0.38 |
| 28 | v5 | active | fmt=2P|phase=mid | 310 | -0.303 | 0.90 | -34% | -9.9 | 0.56 | 0.34 | 0.38 |
| 29 | prod | ships | standing=ahead | 291 | -59.522 | 108.38 | -55% | -9.7 | 0.57 | 0.55 | 0.75 |
| 30 | prod | ships | fmt=2P|standing=ahead | 291 | -59.522 | 108.38 | -55% | -9.7 | 0.57 | 0.55 | 0.75 |
| 31 | v5 | waves | phase=mid | 310 | -0.929 | 2.07 | -45% | -9.4 | 0.53 | 0.61 | 0.77 |
| 32 | v5 | waves | fmt=2P|phase=mid | 310 | -0.929 | 2.07 | -45% | -9.4 | 0.53 | 0.61 | 0.77 |
| 33 | v5 | sources | phase=mid | 310 | -0.929 | 2.07 | -45% | -9.4 | 0.53 | 0.61 | 0.77 |
| 34 | v5 | sources | fmt=2P|phase=mid | 310 | -0.929 | 2.07 | -45% | -9.4 | 0.53 | 0.61 | 0.77 |
| 35 | prod | active | standing=behind | 259 | -0.317 | 0.76 | -41% | -8.7 | 0.54 | 0.38 | 0.44 |
| 36 | prod | active | fmt=2P|standing=behind | 259 | -0.317 | 0.76 | -41% | -8.7 | 0.54 | 0.38 | 0.44 |
| 37 | v5 | ships | all=all | 550 | -49.375 | 90.51 | -55% | -10.0 | 0.42 | 0.50 | 0.75 |
| 38 | v5 | ships | contested=uncontested | 550 | -49.375 | 90.51 | -55% | -10.0 | 0.42 | 0.50 | 0.75 |
| 39 | v5 | ships | fmt=2P | 550 | -49.375 | 90.51 | -55% | -10.0 | 0.42 | 0.50 | 0.75 |
| 40 | v5 | ships | fmt=2P|contested=uncontested | 550 | -49.375 | 90.51 | -55% | -10.0 | 0.42 | 0.50 | 0.75 |
| 41 | v5 | waves | standing=behind | 259 | -0.784 | 1.54 | -51% | -8.2 | 0.51 | 0.49 | 0.63 |
| 42 | v5 | waves | fmt=2P|standing=behind | 259 | -0.784 | 1.54 | -51% | -8.2 | 0.51 | 0.49 | 0.63 |
| 43 | v5 | sources | standing=behind | 259 | -0.784 | 1.54 | -51% | -8.2 | 0.51 | 0.49 | 0.63 |
| 44 | v5 | sources | fmt=2P|standing=behind | 259 | -0.784 | 1.54 | -51% | -8.2 | 0.51 | 0.49 | 0.63 |
| 45 | prod | ships | standing=behind | 259 | -77.857 | 110.32 | -71% | -8.0 | 0.50 | 0.59 | 0.79 |
| 46 | prod | ships | fmt=2P|standing=behind | 259 | -77.857 | 110.32 | -71% | -8.0 | 0.50 | 0.59 | 0.79 |
| 47 | v5 | active | standing=behind | 259 | -0.282 | 0.73 | -39% | -7.7 | 0.48 | 0.36 | 0.43 |
| 48 | v5 | active | fmt=2P|standing=behind | 259 | -0.282 | 0.73 | -39% | -7.7 | 0.48 | 0.36 | 0.43 |
| 49 | prod | neutral_rate | all=all | 264 | -0.154 | 0.26 | -59% | -7.4 | 0.45 | 0.26 | 0.30 |
| 50 | prod | neutral_rate | contested=uncontested | 264 | -0.154 | 0.26 | -59% | -7.4 | 0.45 | 0.26 | 0.30 |
| 51 | prod | neutral_rate | fmt=2P | 264 | -0.154 | 0.26 | -59% | -7.4 | 0.45 | 0.26 | 0.30 |
| 52 | prod | neutral_rate | fmt=2P|contested=uncontested | 264 | -0.154 | 0.26 | -59% | -7.4 | 0.45 | 0.26 | 0.30 |
| 53 | v5 | waves | all=all | 550 | -0.547 | 1.53 | -36% | -8.3 | 0.36 | 0.45 | 0.64 |
| 54 | v5 | waves | contested=uncontested | 550 | -0.547 | 1.53 | -36% | -8.3 | 0.36 | 0.45 | 0.64 |
| 55 | v5 | waves | fmt=2P | 550 | -0.547 | 1.53 | -36% | -8.3 | 0.36 | 0.45 | 0.64 |
| 56 | v5 | waves | fmt=2P|contested=uncontested | 550 | -0.547 | 1.53 | -36% | -8.3 | 0.36 | 0.45 | 0.64 |
| 57 | v5 | sources | all=all | 550 | -0.547 | 1.53 | -36% | -8.3 | 0.36 | 0.45 | 0.64 |
| 58 | v5 | sources | contested=uncontested | 550 | -0.547 | 1.53 | -36% | -8.3 | 0.36 | 0.45 | 0.64 |
| 59 | v5 | sources | fmt=2P | 550 | -0.547 | 1.53 | -36% | -8.3 | 0.36 | 0.45 | 0.64 |
| 60 | v5 | sources | fmt=2P|contested=uncontested | 550 | -0.547 | 1.53 | -36% | -8.3 | 0.36 | 0.45 | 0.64 |
