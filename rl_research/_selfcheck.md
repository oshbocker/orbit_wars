# Clone-Residual Divergence Mine — Findings

_Counterfactual action-diff of top-tier producer-clones vs bare `producer` and our `v5`, on identical observations. Discovery only — gate at n>=100 mirror A/B before shipping._

- Target seats analyzed: **4**; classified **clone** (mean (src,tgt) Jaccard vs producer >= 0.5): **3**; clone decision-turns: **730**.


## Seat roster (clone classification)

| team | rank | rtg | fmt | turns | Jaccard vs producer | med frac (actual/prod) | clone? |
|---|---|---|---|---|---|---|---|
| Oshbocker | 628 | 1139 | 4P | 119 | 0.61 | 1.00/1.00 | ✅ |
| Oshbocker | 628 | 1139 | 4P | 499 | 0.60 | 1.00/1.00 | ✅ |
| Oshbocker | 628 | 1139 | 4P | 112 | 0.60 | 1.00/1.00 | ✅ |
| Oshbocker | 628 | 1139 | 4P | 153 | 0.47 | 1.00/1.00 | — |

## Ranked systematic divergences (clone − baseline)

Positive `mean Δ` = clone does MORE of the axis than the baseline on the same obs. `cons` = fraction of turns sharing the sign; `nz` = fraction of nonzero turns; `eff` = |mean|/sd.

| # | vs | axis | state class | n | mean Δ | cons | nz | eff |
|---|---|---|---|---|---|---|---|---|
| 1 | prod | ships | contested=uncontested | 428 | -207.523 | 0.32 | 0.43 | 0.21 |
| 2 | v5 | ships | contested=uncontested | 428 | -207.523 | 0.32 | 0.43 | 0.21 |
| 3 | prod | ships | standing=ahead | 711 | -124.661 | 0.26 | 0.40 | 0.16 |
| 4 | prod | ships | fmt=4P|standing=ahead | 711 | -124.661 | 0.26 | 0.40 | 0.16 |
| 5 | v5 | ships | standing=ahead | 711 | -124.315 | 0.25 | 0.39 | 0.16 |
| 6 | v5 | ships | fmt=4P|standing=ahead | 711 | -124.315 | 0.25 | 0.39 | 0.16 |
| 7 | prod | ships | all=all | 730 | -121.282 | 0.26 | 0.40 | 0.16 |
| 8 | prod | ships | phase=open | 730 | -121.282 | 0.26 | 0.40 | 0.16 |
| 9 | prod | ships | fmt=4P | 730 | -121.282 | 0.26 | 0.40 | 0.16 |
| 10 | prod | ships | fmt=4P|phase=open | 730 | -121.282 | 0.26 | 0.40 | 0.16 |
| 11 | v5 | ships | all=all | 730 | -120.945 | 0.25 | 0.40 | 0.16 |
| 12 | v5 | ships | phase=open | 730 | -120.945 | 0.25 | 0.40 | 0.16 |
| 13 | v5 | ships | fmt=4P | 730 | -120.945 | 0.25 | 0.40 | 0.16 |
| 14 | v5 | ships | fmt=4P|phase=open | 730 | -120.945 | 0.25 | 0.40 | 0.16 |
| 15 | v5 | tgt_dist | all=all | 84 | -7.446 | 0.43 | 0.58 | 0.34 |
| 16 | v5 | tgt_dist | phase=open | 84 | -7.446 | 0.43 | 0.58 | 0.34 |
| 17 | v5 | tgt_dist | fmt=4P | 84 | -7.446 | 0.43 | 0.58 | 0.34 |
| 18 | v5 | tgt_dist | fmt=4P|phase=open | 84 | -7.446 | 0.43 | 0.58 | 0.34 |
| 19 | v5 | tgt_dist | standing=ahead | 83 | -7.418 | 0.42 | 0.58 | 0.34 |
| 20 | v5 | tgt_dist | fmt=4P|standing=ahead | 83 | -7.418 | 0.42 | 0.58 | 0.34 |
| 21 | prod | tgt_dist | all=all | 85 | -7.148 | 0.42 | 0.59 | 0.33 |
| 22 | prod | tgt_dist | phase=open | 85 | -7.148 | 0.42 | 0.59 | 0.33 |
| 23 | prod | tgt_dist | fmt=4P | 85 | -7.148 | 0.42 | 0.59 | 0.33 |
| 24 | prod | tgt_dist | fmt=4P|phase=open | 85 | -7.148 | 0.42 | 0.59 | 0.33 |
| 25 | prod | tgt_dist | standing=ahead | 84 | -7.117 | 0.42 | 0.58 | 0.33 |
| 26 | prod | tgt_dist | fmt=4P|standing=ahead | 84 | -7.117 | 0.42 | 0.58 | 0.33 |
| 27 | v5 | tgt_dist | contested=contested | 50 | -7.741 | 0.42 | 0.54 | 0.45 |
| 28 | prod | tgt_dist | contested=contested | 51 | -7.238 | 0.41 | 0.55 | 0.42 |
| 29 | v5 | ships | contested=contested | 302 | +1.755 | 0.18 | 0.35 | 0.03 |
| 30 | prod | ships | contested=contested | 302 | +0.940 | 0.18 | 0.35 | 0.02 |
| 31 | prod | waves | contested=uncontested | 428 | -0.355 | 0.31 | 0.40 | 0.39 |
| 32 | v5 | waves | contested=uncontested | 428 | -0.355 | 0.31 | 0.40 | 0.39 |
| 33 | prod | sources | contested=uncontested | 428 | -0.355 | 0.31 | 0.40 | 0.39 |
| 34 | v5 | sources | contested=uncontested | 428 | -0.355 | 0.31 | 0.40 | 0.39 |
| 35 | prod | active | contested=uncontested | 428 | -0.206 | 0.29 | 0.37 | 0.36 |
| 36 | v5 | active | contested=uncontested | 428 | -0.206 | 0.29 | 0.37 | 0.36 |
| 37 | prod | waves | standing=ahead | 711 | -0.203 | 0.25 | 0.37 | 0.23 |
| 38 | prod | waves | fmt=4P|standing=ahead | 711 | -0.203 | 0.25 | 0.37 | 0.23 |
| 39 | prod | sources | standing=ahead | 711 | -0.203 | 0.25 | 0.37 | 0.23 |
| 40 | prod | sources | fmt=4P|standing=ahead | 711 | -0.203 | 0.25 | 0.37 | 0.23 |
| 41 | prod | waves | all=all | 730 | -0.195 | 0.25 | 0.38 | 0.22 |
| 42 | prod | waves | phase=open | 730 | -0.195 | 0.25 | 0.38 | 0.22 |
| 43 | prod | waves | fmt=4P | 730 | -0.195 | 0.25 | 0.38 | 0.22 |
| 44 | prod | waves | fmt=4P|phase=open | 730 | -0.195 | 0.25 | 0.38 | 0.22 |
| 45 | prod | sources | all=all | 730 | -0.195 | 0.25 | 0.38 | 0.22 |
| 46 | prod | sources | phase=open | 730 | -0.195 | 0.25 | 0.38 | 0.22 |
| 47 | prod | sources | fmt=4P | 730 | -0.195 | 0.25 | 0.38 | 0.22 |
| 48 | prod | sources | fmt=4P|phase=open | 730 | -0.195 | 0.25 | 0.38 | 0.22 |
| 49 | v5 | waves | standing=ahead | 711 | -0.197 | 0.25 | 0.37 | 0.22 |
| 50 | v5 | waves | fmt=4P|standing=ahead | 711 | -0.197 | 0.25 | 0.37 | 0.22 |
