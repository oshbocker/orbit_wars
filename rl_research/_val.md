# Clone-Residual Divergence Mine — Findings

_Counterfactual action-diff of top-tier producer-clones vs bare `producer` and our `v5`, on identical observations. Discovery only — gate at n>=100 mirror A/B before shipping._

- Target seats analyzed: **22**; classified **clone** (mean (src,tgt) Jaccard vs producer >= 0.5): **3**; clone decision-turns: **989**.


## Seat roster (clone classification)

| team | rank | rtg | fmt | turns | Jaccard vs producer | med frac (actual/prod) | clone? |
|---|---|---|---|---|---|---|---|
| Jake Will | 1 | 1714 | 4P | 324 | 0.20 | 1.00/1.00 | — |
| Jake Will | 1 | 1714 | 2P | 173 | 0.18 | 1.00/1.00 | — |
| Jake Will | 1 | 1714 | 2P | 499 | 0.14 | 1.00/1.00 | — |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 173 | 0.28 | 1.00/1.00 | — |
| Isaiah @ Tufa Labs | 2 | 1688 | 4P | 324 | 0.19 | 1.00/1.00 | — |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 499 | 0.09 | 0.99/1.00 | — |
| Isaiah @ Tufa Labs | 2 | 1688 | 4P | 499 | 0.08 | 0.99/1.00 | — |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 499 | 0.08 | 1.00/1.00 | — |
| flg | 3 | 1640 | 4P | 499 | 0.85 | 1.00/1.00 | ✅ |
| flg | 3 | 1640 | 4P | 499 | 0.28 | 1.00/1.00 | — |
| flg | 3 | 1640 | 2P | 148 | 0.21 | 0.50/1.00 | — |
| Ender | 4 | 1585 | 4P | 324 | 0.86 | 1.00/1.00 | ✅ |
| Xiangyu Liu | 5 | 1566 | 4P | 166 | 0.59 | 1.00/1.00 | ✅ |
| Xiangyu Liu | 5 | 1566 | 4P | 324 | 0.29 | 1.00/1.00 | — |
| Xiangyu Liu | 5 | 1566 | 4P | 148 | 0.18 | 0.75/1.00 | — |
| Xiangyu Liu | 5 | 1566 | 2P | 499 | 0.13 | 1.00/1.00 | — |
| Xiangyu Liu | 5 | 1566 | 4P | 499 | 0.13 | 1.00/1.00 | — |
| Xiangyu Liu | 5 | 1566 | 2P | 499 | 0.10 | 1.00/1.00 | — |
| Xiangyu Liu | 5 | 1566 | 4P | 499 | 0.10 | 1.00/1.00 | — |
| Hober Malloc | 7 | 1543 | 2P | 102 | 0.33 | 1.00/1.00 | — |
| Hober Malloc | 7 | 1543 | 2P | 499 | 0.10 | 1.00/1.00 | — |
| Vadasz & Ascalon | 9 | 1495 | 2P | 102 | 0.31 | 1.00/1.00 | — |

## Ranked systematic divergences (clone − baseline)

Positive `Δ` = clone does MORE of the axis than the baseline on the SAME obs. `base` = baseline's own mean (for scale); `pct` = Δ as % of base; `t` = t-stat (|t|>=3 kept); `eff` = |Δ|/sd; `cons` = sign-agreement; `nz` = fraction of turns exercising the axis. Ranked by |t|·eff.

| # | vs | axis | state class | n | Δ | base | pct | t | eff | cons | nz |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | v5 | waves | contested=contested | 965 | +0.106 | 0.15 | +73% | +5.0 | 0.16 | 0.11 | 0.17 |
| 2 | v5 | waves | fmt=4P|contested=contested | 965 | +0.106 | 0.15 | +73% | +5.0 | 0.16 | 0.11 | 0.17 |
| 3 | v5 | sources | contested=contested | 965 | +0.106 | 0.15 | +73% | +5.0 | 0.16 | 0.11 | 0.17 |
| 4 | v5 | sources | fmt=4P|contested=contested | 965 | +0.106 | 0.15 | +73% | +5.0 | 0.16 | 0.11 | 0.17 |
| 5 | v5 | waves | all=all | 989 | +0.103 | 0.14 | +73% | +5.0 | 0.16 | 0.11 | 0.17 |
| 6 | v5 | waves | fmt=4P | 989 | +0.103 | 0.14 | +73% | +5.0 | 0.16 | 0.11 | 0.17 |
| 7 | v5 | sources | all=all | 989 | +0.103 | 0.14 | +73% | +5.0 | 0.16 | 0.11 | 0.17 |
| 8 | v5 | sources | fmt=4P | 989 | +0.103 | 0.14 | +73% | +5.0 | 0.16 | 0.11 | 0.17 |
| 9 | v5 | waves | standing=behind | 940 | +0.093 | 0.14 | +67% | +4.4 | 0.14 | 0.10 | 0.16 |
| 10 | v5 | waves | fmt=4P|standing=behind | 940 | +0.093 | 0.14 | +67% | +4.4 | 0.14 | 0.10 | 0.16 |
| 11 | v5 | sources | standing=behind | 940 | +0.093 | 0.14 | +67% | +4.4 | 0.14 | 0.10 | 0.16 |
| 12 | v5 | sources | fmt=4P|standing=behind | 940 | +0.093 | 0.14 | +67% | +4.4 | 0.14 | 0.10 | 0.16 |
| 13 | prod | sources | contested=contested | 965 | +0.093 | 0.16 | +59% | +4.4 | 0.14 | 0.11 | 0.17 |
| 14 | prod | sources | fmt=4P|contested=contested | 965 | +0.093 | 0.16 | +59% | +4.4 | 0.14 | 0.11 | 0.17 |
| 15 | prod | sources | all=all | 989 | +0.091 | 0.15 | +59% | +4.4 | 0.14 | 0.11 | 0.17 |
| 16 | prod | sources | fmt=4P | 989 | +0.091 | 0.15 | +59% | +4.4 | 0.14 | 0.11 | 0.17 |
| 17 | prod | waves | contested=contested | 965 | +0.092 | 0.16 | +58% | +4.3 | 0.14 | 0.11 | 0.17 |
| 18 | prod | waves | fmt=4P|contested=contested | 965 | +0.092 | 0.16 | +58% | +4.3 | 0.14 | 0.11 | 0.17 |
| 19 | prod | waves | all=all | 989 | +0.090 | 0.15 | +58% | +4.3 | 0.14 | 0.11 | 0.17 |
| 20 | prod | waves | fmt=4P | 989 | +0.090 | 0.15 | +58% | +4.3 | 0.14 | 0.11 | 0.17 |
| 21 | prod | sources | standing=behind | 940 | +0.080 | 0.15 | +53% | +3.8 | 0.12 | 0.10 | 0.16 |
| 22 | prod | sources | fmt=4P|standing=behind | 940 | +0.080 | 0.15 | +53% | +3.8 | 0.12 | 0.10 | 0.16 |
| 23 | prod | waves | standing=behind | 940 | +0.079 | 0.15 | +52% | +3.7 | 0.12 | 0.10 | 0.16 |
| 24 | prod | waves | fmt=4P|standing=behind | 940 | +0.079 | 0.15 | +52% | +3.7 | 0.12 | 0.10 | 0.16 |
