# Clone-Residual Divergence Mine — Findings

_Counterfactual action-diff of top-tier producer-family agents vs bare `producer` and our `v5`, on the IDENTICAL observation each agent saw. Discovery only — gate at n>=100 mirror A/B before shipping._

- Target seats analyzed: **111**; classified **posture-clone** (median send-fraction >= 0.9 AND launches/source <= 1.4): **102**; clone decision-turns: **33145**.

- **Headline:** the top tier is producer-family in POSTURE (full-drain, ~1 wave/source) yet its (source,target) SELECTION diverges sharply from bare producer — the residual is in WHAT to attack and FROM WHERE, not how much to send. Tables below quantify where/how.


## Seat roster

`Jaccard` = mean per-turn (src,tgt) overlap with producer's counterfactual (LOW even for full-drain clones = the residual). `wps` = launches per source.

| team | rank | rtg | fmt | turns | med frac | wps | Jaccard vs prod | posture-clone? |
|---|---|---|---|---|---|---|---|---|
| Jake Will | 1 | 1714 | 2P | 173 | 1.00 | 1.00 | 0.08 | ✅ |
| Jake Will | 1 | 1714 | 2P | 499 | 1.00 | 1.00 | 0.07 | ✅ |
| Jake Will | 1 | 1714 | 4P | 324 | 1.00 | 1.00 | 0.06 | ✅ |
| Jake Will | 1 | 1714 | 4P | 286 | 1.00 | 1.00 | 0.04 | ✅ |
| Jake Will | 1 | 1714 | 4P | 499 | 1.00 | 1.00 | 0.01 | ✅ |
| Jake Will | 1 | 1714 | 2P | 123 | 1.00 | 1.00 | 0.11 | ✅ |
| Jake Will | 1 | 1714 | 2P | 499 | 1.00 | 1.00 | 0.04 | ✅ |
| Jake Will | 1 | 1714 | 2P | 245 | 1.00 | 1.00 | 0.08 | ✅ |
| Jake Will | 1 | 1714 | 2P | 126 | 1.00 | 1.00 | 0.09 | ✅ |
| Jake Will | 1 | 1714 | 2P | 150 | 1.00 | 1.00 | 0.10 | ✅ |
| Jake Will | 1 | 1714 | 2P | 499 | 1.00 | 1.00 | 0.05 | ✅ |
| Jake Will | 1 | 1714 | 2P | 499 | 1.00 | 1.00 | 0.10 | ✅ |
| Jake Will | 1 | 1714 | 2P | 499 | 1.00 | 1.00 | 0.04 | ✅ |
| Jake Will | 1 | 1714 | 2P | 267 | 1.00 | 1.00 | 0.07 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 173 | 1.00 | 1.00 | 0.12 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 499 | 0.99 | 1.00 | 0.03 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 4P | 324 | 1.00 | 1.00 | 0.05 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 4P | 499 | 0.99 | 1.00 | 0.03 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 4P | 286 | 1.00 | 1.00 | 0.05 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 131 | 1.00 | 1.00 | 0.07 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 499 | 0.98 | 1.00 | 0.03 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 123 | 1.00 | 1.00 | 0.04 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 499 | 1.00 | 1.00 | 0.03 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 245 | 1.00 | 1.00 | 0.08 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 126 | 1.00 | 1.00 | 0.05 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 150 | 1.00 | 1.00 | 0.09 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 499 | 1.00 | 1.00 | 0.03 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 499 | 1.00 | 1.00 | 0.04 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 124 | 1.00 | 1.00 | 0.11 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 499 | 1.00 | 1.00 | 0.06 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Isaiah @ Tufa Labs | 2 | 1688 | 2P | 267 | 1.00 | 1.00 | 0.07 | ✅ |
| flg | 3 | 1640 | 2P | 148 | 0.50 | 1.00 | 0.04 | — |
| flg | 3 | 1640 | 4P | 499 | 1.00 | 1.00 | 0.07 | ✅ |
| flg | 3 | 1640 | 4P | 499 | 1.00 | 1.00 | 0.05 | ✅ |
| flg | 3 | 1640 | 2P | 151 | 1.00 | 1.00 | 0.06 | ✅ |
| flg | 3 | 1640 | 2P | 499 | 0.67 | 1.00 | 0.05 | — |
| flg | 3 | 1640 | 2P | 499 | 0.50 | 1.00 | 0.06 | — |
| flg | 3 | 1640 | 2P | 124 | 1.00 | 1.00 | 0.13 | ✅ |
| flg | 3 | 1640 | 2P | 153 | 1.00 | 1.00 | 0.05 | ✅ |
| Ender | 4 | 1585 | 4P | 324 | 1.00 | 1.00 | 0.09 | ✅ |
| Ender | 4 | 1585 | 2P | 252 | 1.00 | 1.00 | 0.07 | ✅ |
| Ender | 4 | 1585 | 2P | 131 | 1.00 | 1.00 | 0.13 | ✅ |
| Ender | 4 | 1585 | 4P | 139 | 1.00 | 1.00 | 0.05 | ✅ |
| Ender | 4 | 1585 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Ender | 4 | 1585 | 2P | 118 | 1.00 | 1.00 | 0.06 | ✅ |
| Ender | 4 | 1585 | 2P | 131 | 1.00 | 1.00 | 0.03 | ✅ |
| Ender | 4 | 1585 | 2P | 499 | 1.00 | 1.00 | 0.01 | ✅ |
| Ender | 4 | 1585 | 4P | 499 | 1.00 | 1.00 | 0.04 | ✅ |
| Xiangyu Liu | 5 | 1566 | 2P | 499 | 1.00 | 1.00 | 0.04 | ✅ |
| Xiangyu Liu | 5 | 1566 | 2P | 499 | 1.00 | 1.00 | 0.10 | ✅ |
| Xiangyu Liu | 5 | 1566 | 4P | 499 | 1.00 | 1.00 | 0.07 | ✅ |
| Xiangyu Liu | 5 | 1566 | 4P | 166 | 1.00 | 1.00 | 0.02 | ✅ |
| Xiangyu Liu | 5 | 1566 | 4P | 324 | 1.00 | 1.00 | 0.09 | ✅ |
| Xiangyu Liu | 5 | 1566 | 4P | 499 | 1.00 | 1.00 | 0.07 | ✅ |
| Xiangyu Liu | 5 | 1566 | 4P | 148 | 0.75 | 1.00 | 0.02 | — |
| Xiangyu Liu | 5 | 1566 | 2P | 117 | 1.00 | 1.00 | 0.11 | ✅ |
| Xiangyu Liu | 5 | 1566 | 2P | 228 | 1.00 | 1.00 | 0.07 | ✅ |
| Xiangyu Liu | 5 | 1566 | 4P | 142 | 1.00 | 1.00 | 0.09 | ✅ |
| Xiangyu Liu | 5 | 1566 | 2P | 172 | 1.00 | 1.00 | 0.13 | ✅ |
| Xiangyu Liu | 5 | 1566 | 2P | 499 | 1.00 | 1.00 | 0.04 | ✅ |
| Xiangyu Liu | 5 | 1566 | 2P | 499 | 0.25 | 1.00 | 0.02 | — |
| Xiangyu Liu | 5 | 1566 | 2P | 170 | 1.00 | 1.00 | 0.03 | ✅ |
| Xiangyu Liu | 5 | 1566 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Xiangyu Liu | 5 | 1566 | 2P | 133 | 1.00 | 1.00 | 0.07 | ✅ |
| Xiangyu Liu | 5 | 1566 | 2P | 218 | 1.00 | 1.00 | 0.07 | ✅ |
| Xiangyu Liu | 5 | 1566 | 4P | 171 | 0.75 | 1.00 | 0.02 | — |
| Xiangyu Liu | 5 | 1566 | 2P | 499 | 1.00 | 1.00 | 0.05 | ✅ |
| Xiangyu Liu | 5 | 1566 | 2P | 499 | 1.00 | 1.00 | 0.05 | ✅ |
| Xiangyu Liu | 5 | 1566 | 4P | 243 | 1.00 | 1.00 | 0.09 | ✅ |
| Xiangyu Liu | 5 | 1566 | 2P | 179 | 1.00 | 1.00 | 0.05 | ✅ |
| Xiangyu Liu | 5 | 1566 | 2P | 131 | 1.00 | 1.00 | 0.07 | ✅ |
| Xiangyu Liu | 5 | 1566 | 2P | 499 | 1.00 | 1.00 | 0.04 | ✅ |
| Xiangyu Liu | 5 | 1566 | 4P | 220 | 0.25 | 1.00 | 0.04 | — |
| Xiangyu Liu | 5 | 1566 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Xiangyu Liu | 5 | 1566 | 2P | 499 | 1.00 | 1.00 | 0.04 | ✅ |
| Xiangyu Liu | 5 | 1566 | 2P | 499 | 1.00 | 1.00 | 0.05 | ✅ |
| Xiangyu Liu | 5 | 1566 | 2P | 499 | 1.00 | 1.00 | 0.04 | ✅ |
| Felix M Neumann | 6 | 1556 | 4P | 234 | 0.72 | 1.00 | 0.13 | — |
| Felix M Neumann | 6 | 1556 | 2P | 256 | 0.75 | 1.00 | 0.12 | — |
| Hober Malloc | 7 | 1543 | 2P | 102 | 1.00 | 1.00 | 0.11 | ✅ |
| Hober Malloc | 7 | 1543 | 2P | 499 | 1.00 | 1.00 | 0.01 | ✅ |
| Hober Malloc | 7 | 1543 | 4P | 286 | 1.00 | 1.00 | 0.07 | ✅ |
| Hober Malloc | 7 | 1543 | 2P | 105 | 1.00 | 1.00 | 0.11 | ✅ |
| Hober Malloc | 7 | 1543 | 2P | 118 | 1.00 | 1.00 | 0.03 | ✅ |
| Audun Ljone Henriksen | 8 | 1525 | 2P | 174 | 0.96 | 1.00 | 0.10 | ✅ |
| Audun Ljone Henriksen | 8 | 1525 | 2P | 116 | 0.95 | 1.00 | 0.07 | ✅ |
| Audun Ljone Henriksen | 8 | 1525 | 4P | 173 | 0.97 | 1.00 | 0.09 | ✅ |
| Audun Ljone Henriksen | 8 | 1525 | 4P | 315 | 0.98 | 1.00 | 0.04 | ✅ |
| Vadasz & Ascalon | 9 | 1495 | 2P | 102 | 1.00 | 1.00 | 0.06 | ✅ |
| Vadasz & Ascalon | 9 | 1495 | 2P | 191 | 1.00 | 1.00 | 0.12 | ✅ |
| Vadasz & Ascalon | 9 | 1495 | 4P | 174 | 1.00 | 1.01 | 0.15 | ✅ |
| Vadasz & Ascalon | 9 | 1495 | 4P | 173 | 1.00 | 1.03 | 0.09 | ✅ |
| Vadasz & Ascalon | 9 | 1495 | 2P | 164 | 1.00 | 1.04 | 0.08 | ✅ |
| Vadasz & Ascalon | 9 | 1495 | 4P | 164 | 1.00 | 1.02 | 0.14 | ✅ |
| Yuki Okumura | 10 | 1494 | 2P | 499 | 1.00 | 1.00 | 0.03 | ✅ |
| Yuki Okumura | 10 | 1494 | 2P | 132 | 1.00 | 1.00 | 0.15 | ✅ |
| Yuki Okumura | 10 | 1494 | 2P | 499 | 1.00 | 1.00 | 0.04 | ✅ |
| Yuki Okumura | 10 | 1494 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Yuki Okumura | 10 | 1494 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Yuki Okumura | 10 | 1494 | 2P | 499 | 1.00 | 1.00 | 0.01 | ✅ |
| Yuki Okumura | 10 | 1494 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Yuki Okumura | 10 | 1494 | 2P | 175 | 1.00 | 1.00 | 0.07 | ✅ |
| Yuki Okumura | 10 | 1494 | 4P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Yuki Okumura | 10 | 1494 | 2P | 499 | 1.00 | 1.00 | 0.01 | ✅ |
| Yuki Okumura | 10 | 1494 | 2P | 499 | 1.00 | 1.00 | 0.01 | ✅ |
| Yuki Okumura | 10 | 1494 | 2P | 138 | 1.00 | 1.00 | 0.09 | ✅ |

## Selection divergence (clone vs baseline, same obs)

`pair_prec` = of the clone's launches, fraction producer also makes (low = clone makes launches producer wouldn't = different/extra targets). `pair_rec` = of producer's launches, fraction the clone also makes (low = clone SKIPS launches producer makes). `src_*` = same at source-planet granularity. `tgt_match|src` = on shared source planets, fraction with the SAME target. 1.0 = identical to producer; lower = more divergent.

| vs | metric | state class | n | mean | sd |
|---|---|---|---|---|---|
| prod | pair_prec | all=all | 15823 | 0.150 | 0.305 |
| prod | pair_prec | contested=contested | 2768 | 0.135 | 0.301 |
| prod | pair_prec | contested=uncontested | 13055 | 0.153 | 0.305 |
| prod | pair_prec | fmt=2P | 12524 | 0.153 | 0.305 |
| prod | pair_prec | fmt=2P|contested=uncontested | 12524 | 0.153 | 0.305 |
| prod | pair_prec | fmt=2P|phase=end | 2584 | 0.102 | 0.234 |
| prod | pair_prec | fmt=2P|phase=mid | 7697 | 0.150 | 0.296 |
| prod | pair_prec | fmt=2P|phase=open | 2243 | 0.225 | 0.384 |
| prod | pair_prec | fmt=2P|standing=ahead | 7398 | 0.152 | 0.308 |
| prod | pair_prec | fmt=2P|standing=behind | 5126 | 0.155 | 0.302 |
| prod | pair_prec | fmt=4P | 3299 | 0.137 | 0.302 |
| prod | pair_prec | fmt=4P|contested=contested | 2768 | 0.135 | 0.301 |
| prod | pair_prec | fmt=4P|contested=uncontested | 531 | 0.146 | 0.310 |
| prod | pair_prec | fmt=4P|phase=end | 389 | 0.154 | 0.314 |
| prod | pair_prec | fmt=4P|phase=mid | 2291 | 0.133 | 0.295 |
| prod | pair_prec | fmt=4P|phase=open | 619 | 0.139 | 0.321 |
| prod | pair_prec | fmt=4P|standing=ahead | 1039 | 0.140 | 0.307 |
| prod | pair_prec | fmt=4P|standing=behind | 2260 | 0.135 | 0.300 |
| prod | pair_prec | phase=end | 2973 | 0.109 | 0.247 |
| prod | pair_prec | phase=mid | 9988 | 0.146 | 0.296 |
| prod | pair_prec | phase=open | 2862 | 0.206 | 0.373 |
| prod | pair_prec | standing=ahead | 8437 | 0.151 | 0.308 |
| prod | pair_prec | standing=behind | 7386 | 0.149 | 0.301 |
| prod | pair_rec | all=all | 25826 | 0.068 | 0.201 |
| prod | pair_rec | contested=contested | 3242 | 0.104 | 0.258 |
| prod | pair_rec | contested=uncontested | 22584 | 0.063 | 0.191 |
| prod | pair_rec | fmt=2P | 21258 | 0.064 | 0.192 |
| prod | pair_rec | fmt=2P|contested=uncontested | 21258 | 0.064 | 0.192 |
| prod | pair_rec | fmt=2P|phase=end | 5442 | 0.027 | 0.107 |
| prod | pair_rec | fmt=2P|phase=mid | 13559 | 0.057 | 0.172 |
| prod | pair_rec | fmt=2P|phase=open | 2257 | 0.193 | 0.352 |
| prod | pair_rec | fmt=2P|standing=ahead | 14057 | 0.052 | 0.175 |
| prod | pair_rec | fmt=2P|standing=behind | 7201 | 0.087 | 0.221 |
| prod | pair_rec | fmt=4P | 4568 | 0.087 | 0.238 |
| prod | pair_rec | fmt=4P|contested=contested | 3242 | 0.104 | 0.258 |
| prod | pair_rec | fmt=4P|contested=uncontested | 1326 | 0.046 | 0.174 |
| prod | pair_rec | fmt=4P|phase=end | 756 | 0.055 | 0.175 |
| prod | pair_rec | fmt=4P|phase=mid | 3155 | 0.087 | 0.235 |
| prod | pair_rec | fmt=4P|phase=open | 657 | 0.124 | 0.301 |
| prod | pair_rec | fmt=4P|standing=ahead | 1715 | 0.076 | 0.227 |
| prod | pair_rec | fmt=4P|standing=behind | 2853 | 0.093 | 0.244 |
| prod | pair_rec | phase=end | 6198 | 0.031 | 0.118 |
| prod | pair_rec | phase=mid | 16714 | 0.063 | 0.186 |
| prod | pair_rec | phase=open | 2914 | 0.178 | 0.342 |
| prod | pair_rec | standing=ahead | 15772 | 0.055 | 0.181 |
| prod | pair_rec | standing=behind | 10054 | 0.089 | 0.228 |
| prod | src_prec | all=all | 15823 | 0.339 | 0.404 |
| prod | src_prec | contested=contested | 2768 | 0.315 | 0.406 |
| prod | src_prec | contested=uncontested | 13055 | 0.344 | 0.403 |
| prod | src_prec | fmt=2P | 12524 | 0.345 | 0.402 |
| prod | src_prec | fmt=2P|contested=uncontested | 12524 | 0.345 | 0.402 |
| prod | src_prec | fmt=2P|phase=end | 2584 | 0.287 | 0.367 |
| prod | src_prec | fmt=2P|phase=mid | 7697 | 0.347 | 0.398 |
| prod | src_prec | fmt=2P|phase=open | 2243 | 0.408 | 0.444 |
| prod | src_prec | fmt=2P|standing=ahead | 7398 | 0.344 | 0.407 |
| prod | src_prec | fmt=2P|standing=behind | 5126 | 0.348 | 0.395 |
| prod | src_prec | fmt=4P | 3299 | 0.314 | 0.407 |
| prod | src_prec | fmt=4P|contested=contested | 2768 | 0.315 | 0.406 |
| prod | src_prec | fmt=4P|contested=uncontested | 531 | 0.312 | 0.411 |
| prod | src_prec | fmt=4P|phase=end | 389 | 0.274 | 0.386 |
| prod | src_prec | fmt=4P|phase=mid | 2291 | 0.314 | 0.400 |
| prod | src_prec | fmt=4P|phase=open | 619 | 0.339 | 0.443 |
| prod | src_prec | fmt=4P|standing=ahead | 1039 | 0.345 | 0.415 |
| prod | src_prec | fmt=4P|standing=behind | 2260 | 0.300 | 0.403 |
| prod | src_prec | phase=end | 2973 | 0.285 | 0.369 |
| prod | src_prec | phase=mid | 9988 | 0.339 | 0.399 |
| prod | src_prec | phase=open | 2862 | 0.393 | 0.445 |
| prod | src_prec | standing=ahead | 8437 | 0.344 | 0.408 |
| prod | src_prec | standing=behind | 7386 | 0.333 | 0.398 |
| prod | src_rec | all=all | 25826 | 0.149 | 0.281 |
| prod | src_rec | contested=contested | 3242 | 0.241 | 0.362 |
| prod | src_rec | contested=uncontested | 22584 | 0.135 | 0.264 |
| prod | src_rec | fmt=2P | 21258 | 0.138 | 0.266 |
| prod | src_rec | fmt=2P|contested=uncontested | 21258 | 0.138 | 0.266 |
| prod | src_rec | fmt=2P|phase=end | 5442 | 0.071 | 0.166 |
| prod | src_rec | fmt=2P|phase=mid | 13559 | 0.131 | 0.252 |
| prod | src_rec | fmt=2P|phase=open | 2257 | 0.343 | 0.406 |
| prod | src_rec | fmt=2P|standing=ahead | 14057 | 0.111 | 0.241 |
| prod | src_rec | fmt=2P|standing=behind | 7201 | 0.192 | 0.302 |
| prod | src_rec | fmt=4P | 4568 | 0.198 | 0.337 |
| prod | src_rec | fmt=4P|contested=contested | 3242 | 0.241 | 0.362 |
| prod | src_rec | fmt=4P|contested=uncontested | 1326 | 0.091 | 0.233 |
| prod | src_rec | fmt=4P|phase=end | 756 | 0.102 | 0.234 |
| prod | src_rec | fmt=4P|phase=mid | 3155 | 0.202 | 0.336 |
| prod | src_rec | fmt=4P|phase=open | 657 | 0.286 | 0.403 |
| prod | src_rec | fmt=4P|standing=ahead | 1715 | 0.180 | 0.325 |
| prod | src_rec | fmt=4P|standing=behind | 2853 | 0.208 | 0.343 |
| prod | src_rec | phase=end | 6198 | 0.075 | 0.176 |
| prod | src_rec | phase=mid | 16714 | 0.144 | 0.272 |
| prod | src_rec | phase=open | 2914 | 0.330 | 0.406 |
| prod | src_rec | standing=ahead | 15772 | 0.118 | 0.253 |
| prod | src_rec | standing=behind | 10054 | 0.197 | 0.314 |
| prod | tgt_match|src | all=all | 7930 | 0.428 | 0.465 |
| prod | tgt_match|src | contested=contested | 1210 | 0.420 | 0.472 |
| prod | tgt_match|src | contested=uncontested | 6720 | 0.429 | 0.464 |
| prod | tgt_match|src | fmt=2P | 6496 | 0.428 | 0.463 |
| prod | tgt_match|src | fmt=2P|contested=uncontested | 6496 | 0.428 | 0.463 |
| prod | tgt_match|src | fmt=2P|phase=end | 1260 | 0.369 | 0.450 |
| prod | tgt_match|src | fmt=2P|phase=mid | 4099 | 0.416 | 0.457 |
| prod | tgt_match|src | fmt=2P|phase=open | 1137 | 0.533 | 0.480 |
| prod | tgt_match|src | fmt=2P|standing=ahead | 3750 | 0.424 | 0.462 |
| prod | tgt_match|src | fmt=2P|standing=behind | 2746 | 0.432 | 0.465 |
| prod | tgt_match|src | fmt=4P | 1434 | 0.427 | 0.474 |
| prod | tgt_match|src | fmt=4P|contested=contested | 1210 | 0.420 | 0.472 |
| prod | tgt_match|src | fmt=4P|contested=uncontested | 224 | 0.469 | 0.481 |
| prod | tgt_match|src | fmt=4P|phase=end | 153 | 0.540 | 0.484 |
| prod | tgt_match|src | fmt=4P|phase=mid | 1027 | 0.416 | 0.469 |
| prod | tgt_match|src | fmt=4P|phase=open | 254 | 0.407 | 0.478 |
| prod | tgt_match|src | fmt=4P|standing=ahead | 500 | 0.387 | 0.462 |
| prod | tgt_match|src | fmt=4P|standing=behind | 934 | 0.449 | 0.479 |
| prod | tgt_match|src | phase=end | 1413 | 0.388 | 0.457 |
| prod | tgt_match|src | phase=mid | 5126 | 0.416 | 0.460 |
| prod | tgt_match|src | phase=open | 1391 | 0.510 | 0.482 |
| prod | tgt_match|src | standing=ahead | 4250 | 0.420 | 0.462 |
| prod | tgt_match|src | standing=behind | 3680 | 0.437 | 0.469 |
| v5 | pair_prec | all=all | 15823 | 0.143 | 0.300 |
| v5 | pair_prec | contested=contested | 2768 | 0.131 | 0.298 |
| v5 | pair_prec | contested=uncontested | 13055 | 0.145 | 0.301 |
| v5 | pair_prec | fmt=2P | 12524 | 0.145 | 0.300 |
| v5 | pair_prec | fmt=2P|contested=uncontested | 12524 | 0.145 | 0.300 |
| v5 | pair_prec | fmt=2P|phase=end | 2584 | 0.088 | 0.225 |
| v5 | pair_prec | fmt=2P|phase=mid | 7697 | 0.144 | 0.292 |
| v5 | pair_prec | fmt=2P|phase=open | 2243 | 0.218 | 0.378 |
| v5 | pair_prec | fmt=2P|standing=ahead | 7398 | 0.147 | 0.304 |
| v5 | pair_prec | fmt=2P|standing=behind | 5126 | 0.144 | 0.295 |
| v5 | pair_prec | fmt=4P | 3299 | 0.132 | 0.300 |
| v5 | pair_prec | fmt=4P|contested=contested | 2768 | 0.131 | 0.298 |
| v5 | pair_prec | fmt=4P|contested=uncontested | 531 | 0.141 | 0.309 |
| v5 | pair_prec | fmt=4P|phase=end | 389 | 0.139 | 0.301 |
| v5 | pair_prec | fmt=4P|phase=mid | 2291 | 0.128 | 0.292 |
| v5 | pair_prec | fmt=4P|phase=open | 619 | 0.144 | 0.327 |
| v5 | pair_prec | fmt=4P|standing=ahead | 1039 | 0.134 | 0.301 |
| v5 | pair_prec | fmt=4P|standing=behind | 2260 | 0.132 | 0.299 |
| v5 | pair_prec | phase=end | 2973 | 0.095 | 0.237 |
| v5 | pair_prec | phase=mid | 9988 | 0.140 | 0.292 |
| v5 | pair_prec | phase=open | 2862 | 0.202 | 0.369 |
| v5 | pair_prec | standing=ahead | 8437 | 0.145 | 0.304 |
| v5 | pair_prec | standing=behind | 7386 | 0.140 | 0.296 |
| v5 | pair_rec | all=all | 24537 | 0.074 | 0.214 |
| v5 | pair_rec | contested=contested | 3044 | 0.113 | 0.273 |
| v5 | pair_rec | contested=uncontested | 21493 | 0.069 | 0.204 |
| v5 | pair_rec | fmt=2P | 20422 | 0.070 | 0.205 |
| v5 | pair_rec | fmt=2P|contested=uncontested | 20422 | 0.070 | 0.205 |
| v5 | pair_rec | fmt=2P|phase=end | 5109 | 0.030 | 0.119 |
| v5 | pair_rec | fmt=2P|phase=mid | 13150 | 0.063 | 0.186 |
| v5 | pair_rec | fmt=2P|phase=open | 2163 | 0.205 | 0.361 |
| v5 | pair_rec | fmt=2P|standing=ahead | 13798 | 0.056 | 0.184 |
| v5 | pair_rec | fmt=2P|standing=behind | 6624 | 0.097 | 0.239 |
| v5 | pair_rec | fmt=4P | 4115 | 0.097 | 0.255 |
| v5 | pair_rec | fmt=4P|contested=contested | 3044 | 0.113 | 0.273 |
| v5 | pair_rec | fmt=4P|contested=uncontested | 1071 | 0.054 | 0.187 |
| v5 | pair_rec | fmt=4P|phase=end | 586 | 0.064 | 0.184 |
| v5 | pair_rec | fmt=4P|phase=mid | 2918 | 0.095 | 0.251 |
| v5 | pair_rec | fmt=4P|phase=open | 611 | 0.141 | 0.319 |
| v5 | pair_rec | fmt=4P|standing=ahead | 1665 | 0.078 | 0.232 |
| v5 | pair_rec | fmt=4P|standing=behind | 2450 | 0.111 | 0.269 |
| v5 | pair_rec | phase=end | 5695 | 0.033 | 0.128 |
| v5 | pair_rec | phase=mid | 16068 | 0.069 | 0.200 |
| v5 | pair_rec | phase=open | 2774 | 0.191 | 0.353 |
| v5 | pair_rec | standing=ahead | 15463 | 0.059 | 0.190 |
| v5 | pair_rec | standing=behind | 9074 | 0.101 | 0.248 |
| v5 | src_prec | all=all | 15823 | 0.297 | 0.391 |
| v5 | src_prec | contested=contested | 2768 | 0.286 | 0.395 |
| v5 | src_prec | contested=uncontested | 13055 | 0.299 | 0.390 |
| v5 | src_prec | fmt=2P | 12524 | 0.299 | 0.389 |
| v5 | src_prec | fmt=2P|contested=uncontested | 12524 | 0.299 | 0.389 |
| v5 | src_prec | fmt=2P|phase=end | 2584 | 0.219 | 0.340 |
| v5 | src_prec | fmt=2P|phase=mid | 7697 | 0.303 | 0.385 |
| v5 | src_prec | fmt=2P|phase=open | 2243 | 0.377 | 0.438 |
| v5 | src_prec | fmt=2P|standing=ahead | 7398 | 0.308 | 0.397 |
| v5 | src_prec | fmt=2P|standing=behind | 5126 | 0.286 | 0.377 |
| v5 | src_prec | fmt=4P | 3299 | 0.289 | 0.397 |
| v5 | src_prec | fmt=4P|contested=contested | 2768 | 0.286 | 0.395 |
| v5 | src_prec | fmt=4P|contested=uncontested | 531 | 0.301 | 0.410 |
| v5 | src_prec | fmt=4P|phase=end | 389 | 0.249 | 0.375 |
| v5 | src_prec | fmt=4P|phase=mid | 2291 | 0.289 | 0.390 |
| v5 | src_prec | fmt=4P|phase=open | 619 | 0.310 | 0.434 |
| v5 | src_prec | fmt=4P|standing=ahead | 1039 | 0.323 | 0.409 |
| v5 | src_prec | fmt=4P|standing=behind | 2260 | 0.273 | 0.391 |
| v5 | src_prec | phase=end | 2973 | 0.223 | 0.345 |
| v5 | src_prec | phase=mid | 9988 | 0.300 | 0.386 |
| v5 | src_prec | phase=open | 2862 | 0.363 | 0.438 |
| v5 | src_prec | standing=ahead | 8437 | 0.310 | 0.399 |
| v5 | src_prec | standing=behind | 7386 | 0.282 | 0.382 |
| v5 | src_rec | all=all | 24537 | 0.152 | 0.292 |
| v5 | src_rec | contested=contested | 3044 | 0.249 | 0.373 |
| v5 | src_rec | contested=uncontested | 21493 | 0.138 | 0.275 |
| v5 | src_rec | fmt=2P | 20422 | 0.140 | 0.276 |
| v5 | src_rec | fmt=2P|contested=uncontested | 20422 | 0.140 | 0.276 |
| v5 | src_rec | fmt=2P|phase=end | 5109 | 0.071 | 0.183 |
| v5 | src_rec | fmt=2P|phase=mid | 13150 | 0.133 | 0.262 |
| v5 | src_rec | fmt=2P|phase=open | 2163 | 0.348 | 0.413 |
| v5 | src_rec | fmt=2P|standing=ahead | 13798 | 0.114 | 0.251 |
| v5 | src_rec | fmt=2P|standing=behind | 6624 | 0.194 | 0.317 |
| v5 | src_rec | fmt=4P | 4115 | 0.212 | 0.351 |
| v5 | src_rec | fmt=4P|contested=contested | 3044 | 0.249 | 0.373 |
| v5 | src_rec | fmt=4P|contested=uncontested | 1071 | 0.109 | 0.251 |
| v5 | src_rec | fmt=4P|phase=end | 586 | 0.121 | 0.248 |
| v5 | src_rec | fmt=4P|phase=mid | 2918 | 0.214 | 0.350 |
| v5 | src_rec | fmt=4P|phase=open | 611 | 0.294 | 0.412 |
| v5 | src_rec | fmt=4P|standing=ahead | 1665 | 0.182 | 0.333 |
| v5 | src_rec | fmt=4P|standing=behind | 2450 | 0.233 | 0.361 |
| v5 | src_rec | phase=end | 5695 | 0.076 | 0.191 |
| v5 | src_rec | phase=mid | 16068 | 0.147 | 0.282 |
| v5 | src_rec | phase=open | 2774 | 0.336 | 0.413 |
| v5 | src_rec | standing=ahead | 15463 | 0.121 | 0.262 |
| v5 | src_rec | standing=behind | 9074 | 0.205 | 0.330 |
| v5 | tgt_match|src | all=all | 7038 | 0.466 | 0.473 |
| v5 | tgt_match|src | contested=contested | 1121 | 0.444 | 0.479 |
| v5 | tgt_match|src | contested=uncontested | 5917 | 0.470 | 0.472 |
| v5 | tgt_match|src | fmt=2P | 5702 | 0.470 | 0.471 |
| v5 | tgt_match|src | fmt=2P|contested=uncontested | 5702 | 0.470 | 0.471 |
| v5 | tgt_match|src | fmt=2P|phase=end | 975 | 0.413 | 0.465 |
| v5 | tgt_match|src | fmt=2P|phase=mid | 3665 | 0.459 | 0.467 |
| v5 | tgt_match|src | fmt=2P|phase=open | 1062 | 0.564 | 0.478 |
| v5 | tgt_match|src | fmt=2P|standing=ahead | 3392 | 0.461 | 0.470 |
| v5 | tgt_match|src | fmt=2P|standing=behind | 2310 | 0.484 | 0.474 |
| v5 | tgt_match|src | fmt=4P | 1336 | 0.447 | 0.479 |
| v5 | tgt_match|src | fmt=4P|contested=contested | 1121 | 0.444 | 0.479 |
| v5 | tgt_match|src | fmt=4P|contested=uncontested | 215 | 0.461 | 0.482 |
| v5 | tgt_match|src | fmt=4P|phase=end | 141 | 0.539 | 0.488 |
| v5 | tgt_match|src | fmt=4P|phase=mid | 961 | 0.431 | 0.475 |
| v5 | tgt_match|src | fmt=4P|phase=open | 234 | 0.458 | 0.485 |
| v5 | tgt_match|src | fmt=4P|standing=ahead | 471 | 0.396 | 0.466 |
| v5 | tgt_match|src | fmt=4P|standing=behind | 865 | 0.475 | 0.484 |
| v5 | tgt_match|src | phase=end | 1116 | 0.429 | 0.470 |
| v5 | tgt_match|src | phase=mid | 4626 | 0.453 | 0.469 |
| v5 | tgt_match|src | phase=open | 1296 | 0.545 | 0.481 |
| v5 | tgt_match|src | standing=ahead | 3863 | 0.453 | 0.470 |
| v5 | tgt_match|src | standing=behind | 3175 | 0.482 | 0.477 |

## Ranked behavioural divergences (clone − baseline)

Positive `Δ` = clone does MORE of the axis than the baseline on the SAME obs. `base` = baseline's own mean (for scale); `pct` = Δ as % of base; `t` = t-stat (|t|>=3 kept); `eff` = |Δ|/sd; `cons` = sign-agreement; `nz` = fraction of turns exercising the axis. Ranked by |t|·eff.

| # | vs | axis | state class | n | Δ | base | pct | t | eff | cons | nz |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | prod | waves | fmt=2P|standing=ahead | 16579 | -3.102 | 4.22 | -73% | -118.2 | 0.92 | 0.73 | 0.82 |
| 2 | prod | sources | fmt=2P|standing=ahead | 16579 | -3.087 | 4.21 | -73% | -118.0 | 0.92 | 0.73 | 0.82 |
| 3 | prod | waves | standing=ahead | 18976 | -2.873 | 3.95 | -73% | -119.0 | 0.86 | 0.70 | 0.81 |
| 4 | prod | sources | standing=ahead | 18976 | -2.860 | 3.94 | -73% | -118.8 | 0.86 | 0.70 | 0.81 |
| 5 | prod | waves | fmt=2P|phase=end | 5662 | -3.762 | 5.09 | -74% | -90.1 | 1.20 | 0.85 | 0.93 |
| 6 | prod | sources | fmt=2P|phase=end | 5662 | -3.740 | 5.07 | -74% | -89.9 | 1.19 | 0.85 | 0.93 |
| 7 | prod | waves | contested=uncontested | 27121 | -2.369 | 3.53 | -67% | -123.6 | 0.75 | 0.67 | 0.81 |
| 8 | prod | sources | contested=uncontested | 27121 | -2.358 | 3.52 | -67% | -123.4 | 0.75 | 0.67 | 0.81 |
| 9 | prod | waves | fmt=2P | 25310 | -2.434 | 3.64 | -67% | -120.7 | 0.76 | 0.67 | 0.81 |
| 10 | prod | waves | fmt=2P|contested=uncontested | 25310 | -2.434 | 3.64 | -67% | -120.7 | 0.76 | 0.67 | 0.81 |
| 11 | prod | sources | fmt=2P | 25310 | -2.423 | 3.63 | -67% | -120.4 | 0.76 | 0.67 | 0.81 |
| 12 | prod | sources | fmt=2P|contested=uncontested | 25310 | -2.423 | 3.63 | -67% | -120.4 | 0.76 | 0.67 | 0.81 |
| 13 | prod | waves | phase=end | 6854 | -3.296 | 4.49 | -73% | -87.0 | 1.05 | 0.79 | 0.87 |
| 14 | prod | sources | phase=end | 6854 | -3.277 | 4.47 | -73% | -86.9 | 1.05 | 0.79 | 0.87 |
| 15 | prod | waves | all=all | 33145 | -1.992 | 3.09 | -64% | -119.3 | 0.66 | 0.61 | 0.76 |
| 16 | prod | sources | all=all | 33145 | -1.983 | 3.08 | -64% | -119.1 | 0.65 | 0.61 | 0.76 |
| 17 | v5 | waves | fmt=2P|standing=ahead | 16579 | -2.392 | 3.51 | -68% | -100.1 | 0.78 | 0.67 | 0.80 |
| 18 | v5 | sources | fmt=2P|standing=ahead | 16579 | -2.385 | 3.50 | -68% | -100.0 | 0.78 | 0.67 | 0.80 |
| 19 | prod | waves | fmt=2P|phase=mid | 14968 | -2.618 | 3.94 | -67% | -96.8 | 0.79 | 0.73 | 0.87 |
| 20 | prod | sources | fmt=2P|phase=mid | 14968 | -2.607 | 3.92 | -66% | -96.5 | 0.79 | 0.73 | 0.87 |
| 21 | v5 | waves | standing=ahead | 18976 | -2.232 | 3.31 | -67% | -101.6 | 0.74 | 0.65 | 0.79 |
| 22 | v5 | sources | standing=ahead | 18976 | -2.226 | 3.31 | -67% | -101.5 | 0.74 | 0.65 | 0.79 |
| 23 | prod | waves | phase=mid | 20171 | -2.091 | 3.28 | -64% | -95.1 | 0.67 | 0.64 | 0.80 |
| 24 | prod | sources | phase=mid | 20171 | -2.083 | 3.27 | -64% | -94.9 | 0.67 | 0.64 | 0.80 |
| 25 | prod | active | fmt=2P|standing=ahead | 16579 | -0.402 | 0.85 | -47% | -90.5 | 0.70 | 0.44 | 0.49 |
| 26 | prod | active | fmt=2P|phase=end | 5662 | -0.505 | 0.96 | -53% | -70.6 | 0.94 | 0.52 | 0.54 |
| 27 | v5 | waves | contested=uncontested | 27121 | -1.725 | 2.89 | -60% | -98.8 | 0.60 | 0.59 | 0.77 |
| 28 | prod | active | standing=ahead | 18976 | -0.387 | 0.83 | -47% | -91.4 | 0.66 | 0.44 | 0.49 |
| 29 | v5 | sources | contested=uncontested | 27121 | -1.720 | 2.88 | -60% | -98.7 | 0.60 | 0.59 | 0.77 |
| 30 | prod | active | contested=uncontested | 27121 | -0.351 | 0.83 | -42% | -98.4 | 0.60 | 0.41 | 0.47 |
| 31 | v5 | waves | fmt=2P | 25310 | -1.762 | 2.97 | -59% | -96.0 | 0.60 | 0.60 | 0.79 |
| 32 | v5 | waves | fmt=2P|contested=uncontested | 25310 | -1.762 | 2.97 | -59% | -96.0 | 0.60 | 0.60 | 0.79 |
| 33 | v5 | sources | fmt=2P | 25310 | -1.757 | 2.97 | -59% | -95.9 | 0.60 | 0.60 | 0.79 |
| 34 | v5 | sources | fmt=2P|contested=uncontested | 25310 | -1.757 | 2.97 | -59% | -95.9 | 0.60 | 0.60 | 0.79 |
| 35 | prod | active | phase=end | 6854 | -0.471 | 0.90 | -52% | -71.0 | 0.86 | 0.50 | 0.52 |
| 36 | prod | active | fmt=2P | 25310 | -0.345 | 0.84 | -41% | -93.9 | 0.59 | 0.40 | 0.46 |
| 37 | prod | active | fmt=2P|contested=uncontested | 25310 | -0.345 | 0.84 | -41% | -93.9 | 0.59 | 0.40 | 0.46 |
| 38 | prod | active | fmt=2P|phase=mid | 14968 | -0.392 | 0.91 | -43% | -83.7 | 0.68 | 0.44 | 0.48 |
| 39 | v5 | active | fmt=2P|standing=ahead | 16579 | -0.386 | 0.83 | -46% | -85.0 | 0.66 | 0.44 | 0.49 |
| 40 | v5 | active | standing=ahead | 18976 | -0.370 | 0.81 | -45% | -85.6 | 0.62 | 0.43 | 0.49 |
| 41 | v5 | waves | all=all | 33145 | -1.443 | 2.54 | -57% | -95.7 | 0.53 | 0.54 | 0.73 |
| 42 | v5 | sources | all=all | 33145 | -1.439 | 2.54 | -57% | -95.6 | 0.53 | 0.54 | 0.73 |
| 43 | v5 | waves | fmt=2P|phase=mid | 14968 | -1.977 | 3.29 | -60% | -79.9 | 0.65 | 0.67 | 0.85 |
| 44 | prod | ships | standing=ahead | 18976 | -1213.291 | 1263.79 | -96% | -84.0 | 0.61 | 0.75 | 0.86 |
| 45 | v5 | sources | fmt=2P|phase=mid | 14968 | -1.971 | 3.29 | -60% | -79.8 | 0.65 | 0.67 | 0.85 |
| 46 | prod | ships | fmt=2P|standing=ahead | 16579 | -1265.541 | 1316.99 | -96% | -81.5 | 0.63 | 0.77 | 0.87 |
| 47 | prod | ships | contested=uncontested | 27121 | -954.916 | 1005.08 | -95% | -89.8 | 0.55 | 0.74 | 0.87 |
| 48 | prod | active | all=all | 33145 | -0.302 | 0.78 | -39% | -93.3 | 0.51 | 0.37 | 0.44 |
| 49 | v5 | ships | standing=ahead | 18976 | -1086.734 | 1137.23 | -96% | -81.9 | 0.59 | 0.72 | 0.85 |
| 50 | prod | ships | fmt=2P | 25310 | -942.806 | 994.07 | -95% | -86.8 | 0.55 | 0.75 | 0.88 |
| 51 | prod | ships | fmt=2P|contested=uncontested | 25310 | -942.806 | 994.07 | -95% | -86.8 | 0.55 | 0.75 | 0.88 |
| 52 | v5 | ships | fmt=2P|standing=ahead | 16579 | -1136.983 | 1188.43 | -96% | -79.3 | 0.62 | 0.75 | 0.86 |
| 53 | prod | ships | fmt=2P|phase=end | 5662 | -2019.310 | 2087.68 | -97% | -62.0 | 0.82 | 0.93 | 0.98 |
| 54 | prod | active | phase=mid | 20171 | -0.333 | 0.83 | -40% | -81.6 | 0.57 | 0.39 | 0.45 |
| 55 | v5 | waves | fmt=2P|phase=end | 5662 | -2.548 | 3.88 | -66% | -61.3 | 0.81 | 0.73 | 0.88 |
| 56 | v5 | sources | fmt=2P|phase=end | 5662 | -2.540 | 3.87 | -66% | -61.2 | 0.81 | 0.73 | 0.88 |
| 57 | prod | ships | all=all | 33145 | -796.353 | 842.80 | -94% | -89.2 | 0.49 | 0.68 | 0.83 |
| 58 | v5 | active | fmt=2P|phase=mid | 14968 | -0.364 | 0.88 | -41% | -74.9 | 0.61 | 0.43 | 0.49 |
| 59 | v5 | ships | contested=uncontested | 27121 | -824.136 | 874.30 | -94% | -84.7 | 0.51 | 0.69 | 0.85 |
| 60 | v5 | waves | phase=mid | 20171 | -1.582 | 2.77 | -57% | -79.1 | 0.56 | 0.58 | 0.77 |
