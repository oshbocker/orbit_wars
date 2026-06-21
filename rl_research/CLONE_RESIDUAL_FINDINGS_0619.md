# Clone-Residual Divergence Mine — Findings

_Counterfactual action-diff of top-tier producer-family agents vs bare `producer` and our `v5`, on the IDENTICAL observation each agent saw. Discovery only — gate at n>=100 mirror A/B before shipping._

- Target seats analyzed: **113**; classified **posture-clone** (median send-fraction >= 0.9 AND launches/source <= 1.4): **106**; clone decision-turns: **29577**.

- **Headline:** the top tier is producer-family in POSTURE (full-drain, ~1 wave/source) yet its (source,target) SELECTION diverges sharply from bare producer — the residual is in WHAT to attack and FROM WHERE, not how much to send. Tables below quantify where/how.


## Seat roster

`Jaccard` = mean per-turn (src,tgt) overlap with producer's counterfactual (LOW even for full-drain clones = the residual). `wps` = launches per source.

| team | rank | rtg | fmt | turns | med frac | wps | Jaccard vs prod | posture-clone? |
|---|---|---|---|---|---|---|---|---|
| Isaiah @ Tufa Labs | 1 | 1704 | 2P | 499 | 1.00 | 1.00 | 0.05 | ✅ |
| Isaiah @ Tufa Labs | 1 | 1704 | 2P | 137 | 1.00 | 1.00 | 0.07 | ✅ |
| Isaiah @ Tufa Labs | 1 | 1704 | 2P | 499 | 1.00 | 1.00 | 0.00 | ✅ |
| Isaiah @ Tufa Labs | 1 | 1704 | 2P | 176 | 1.00 | 1.00 | 0.04 | ✅ |
| Isaiah @ Tufa Labs | 1 | 1704 | 2P | 499 | 0.54 | 1.00 | 0.02 | — |
| Isaiah @ Tufa Labs | 1 | 1704 | 2P | 499 | 1.00 | 1.00 | 0.05 | ✅ |
| Isaiah @ Tufa Labs | 1 | 1704 | 2P | 157 | 1.00 | 1.00 | 0.10 | ✅ |
| Isaiah @ Tufa Labs | 1 | 1704 | 2P | 499 | 1.00 | 1.00 | 0.05 | ✅ |
| Isaiah @ Tufa Labs | 1 | 1704 | 2P | 499 | 0.99 | 1.00 | 0.04 | ✅ |
| Isaiah @ Tufa Labs | 1 | 1704 | 2P | 499 | 1.00 | 1.00 | 0.01 | ✅ |
| Isaiah @ Tufa Labs | 1 | 1704 | 2P | 499 | 1.00 | 1.00 | 0.03 | ✅ |
| Isaiah @ Tufa Labs | 1 | 1704 | 2P | 499 | 0.54 | 1.00 | 0.02 | — |
| Isaiah @ Tufa Labs | 1 | 1704 | 2P | 166 | 1.00 | 1.00 | 0.06 | ✅ |
| Isaiah @ Tufa Labs | 1 | 1704 | 2P | 499 | 1.00 | 1.00 | 0.00 | ✅ |
| Isaiah @ Tufa Labs | 1 | 1704 | 2P | 171 | 1.00 | 1.00 | 0.10 | ✅ |
| Isaiah @ Tufa Labs | 1 | 1704 | 2P | 499 | 1.00 | 1.00 | 0.05 | ✅ |
| Jake Will | 2 | 1669 | 2P | 176 | 1.00 | 1.00 | 0.07 | ✅ |
| Jake Will | 2 | 1669 | 2P | 218 | 1.00 | 1.00 | 0.11 | ✅ |
| Jake Will | 2 | 1669 | 2P | 157 | 1.00 | 1.00 | 0.09 | ✅ |
| Jake Will | 2 | 1669 | 2P | 270 | 1.00 | 1.00 | 0.11 | ✅ |
| Jake Will | 2 | 1669 | 2P | 171 | 1.00 | 1.00 | 0.12 | ✅ |
| Jake Will | 2 | 1669 | 2P | 499 | 1.00 | 1.00 | 0.09 | ✅ |
| Xiangyu Liu | 3 | 1606 | 2P | 128 | 1.00 | 1.00 | 0.15 | ✅ |
| Xiangyu Liu | 3 | 1606 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Xiangyu Liu | 3 | 1606 | 2P | 218 | 1.00 | 1.00 | 0.08 | ✅ |
| Xiangyu Liu | 3 | 1606 | 4P | 499 | 0.33 | 1.00 | 0.02 | — |
| Xiangyu Liu | 3 | 1606 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Xiangyu Liu | 3 | 1606 | 2P | 499 | 1.00 | 1.00 | 0.07 | ✅ |
| Xiangyu Liu | 3 | 1606 | 4P | 328 | 1.00 | 1.00 | 0.07 | ✅ |
| Xiangyu Liu | 3 | 1606 | 4P | 175 | 0.33 | 1.00 | 0.07 | — |
| Xiangyu Liu | 3 | 1606 | 4P | 223 | 1.00 | 1.00 | 0.07 | ✅ |
| Xiangyu Liu | 3 | 1606 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Xiangyu Liu | 3 | 1606 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Xiangyu Liu | 3 | 1606 | 2P | 166 | 1.00 | 1.00 | 0.09 | ✅ |
| Xiangyu Liu | 3 | 1606 | 4P | 499 | 1.00 | 1.00 | 0.00 | ✅ |
| Hober Malloc | 4 | 1605 | 2P | 499 | 1.00 | 1.00 | 0.08 | ✅ |
| Hober Malloc | 4 | 1605 | 2P | 137 | 1.00 | 1.00 | 0.11 | ✅ |
| Hober Malloc | 4 | 1605 | 2P | 499 | 1.00 | 1.00 | 0.09 | ✅ |
| Hober Malloc | 4 | 1605 | 2P | 499 | 1.00 | 1.00 | 0.09 | ✅ |
| Hober Malloc | 4 | 1605 | 4P | 108 | 1.00 | 1.00 | 0.05 | ✅ |
| flg | 5 | 1590 | 2P | 139 | 0.50 | 1.00 | 0.08 | — |
| flg | 5 | 1590 | 2P | 176 | 0.47 | 1.00 | 0.11 | — |
| flg | 5 | 1590 | 2P | 499 | 1.00 | 1.00 | 0.11 | ✅ |
| flg | 5 | 1590 | 2P | 211 | 1.00 | 1.00 | 0.12 | ✅ |
| flg | 5 | 1590 | 2P | 499 | 0.50 | 1.00 | 0.04 | — |
| Boey | 6 | 1533 | 2P | 499 | 1.00 | 1.00 | 0.01 | ✅ |
| Boey | 6 | 1533 | 4P | 328 | 1.00 | 1.03 | 0.05 | ✅ |
| Boey | 6 | 1533 | 4P | 397 | 1.00 | 1.00 | 0.07 | ✅ |
| Boey | 6 | 1533 | 4P | 499 | 1.00 | 1.00 | 0.05 | ✅ |
| Boey | 6 | 1533 | 4P | 499 | 1.00 | 1.00 | 0.05 | ✅ |
| Felix M Neumann | 7 | 1529 | 2P | 135 | 1.00 | 1.00 | 0.15 | ✅ |
| Felix M Neumann | 7 | 1529 | 2P | 270 | 1.00 | 1.00 | 0.09 | ✅ |
| moriiiiiiiiim | 8 | 1518 | 4P | 499 | 1.00 | 1.00 | 0.00 | ✅ |
| moriiiiiiiiim | 8 | 1518 | 4P | 123 | 1.00 | 1.00 | 0.08 | ✅ |
| moriiiiiiiiim | 8 | 1518 | 4P | 499 | 1.00 | 1.00 | 0.04 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 2P | 193 | 1.00 | 1.00 | 0.10 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 2P | 155 | 1.00 | 1.00 | 0.13 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 4P | 171 | 1.00 | 1.00 | 0.22 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 4P | 109 | 1.00 | 1.04 | 0.15 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 4P | 236 | 1.00 | 1.02 | 0.12 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 4P | 241 | 1.00 | 1.03 | 0.02 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 2P | 162 | 1.00 | 1.00 | 0.13 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 2P | 127 | 1.00 | 1.01 | 0.12 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 4P | 208 | 1.00 | 1.02 | 0.08 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 4P | 499 | 1.00 | 1.05 | 0.13 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 2P | 121 | 1.00 | 1.02 | 0.08 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 4P | 128 | 1.00 | 1.00 | 0.06 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 4P | 175 | 1.00 | 1.01 | 0.14 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 4P | 223 | 1.00 | 1.02 | 0.06 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 2P | 155 | 1.00 | 1.00 | 0.14 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 2P | 177 | 1.00 | 1.01 | 0.12 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 2P | 118 | 1.00 | 1.02 | 0.11 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 2P | 162 | 1.00 | 1.00 | 0.06 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 2P | 170 | 1.00 | 1.02 | 0.11 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 4P | 499 | 1.00 | 1.01 | 0.13 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 2P | 126 | 1.00 | 1.02 | 0.13 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 4P | 165 | 1.00 | 1.02 | 0.11 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 2P | 146 | 1.00 | 1.00 | 0.08 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 4P | 499 | 1.00 | 1.01 | 0.06 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 4P | 164 | 1.00 | 1.03 | 0.13 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 4P | 178 | 1.00 | 1.01 | 0.13 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 4P | 108 | 0.93 | 1.00 | 0.20 | ✅ |
| Vadasz & Ascalon | 9 | 1512 | 2P | 183 | 1.00 | 1.00 | 0.06 | ✅ |
| M & J & M.ver2 | 10 | 1510 | 2P | 193 | 1.00 | 1.00 | 0.06 | ✅ |
| M & J & M.ver2 | 10 | 1510 | 2P | 499 | 1.00 | 1.00 | 0.03 | ✅ |
| M & J & M.ver2 | 10 | 1510 | 2P | 192 | 1.00 | 1.00 | 0.08 | ✅ |
| M & J & M.ver2 | 10 | 1510 | 4P | 193 | 1.00 | 1.00 | 0.11 | ✅ |
| M & J & M.ver2 | 10 | 1510 | 2P | 146 | 1.00 | 1.00 | 0.14 | ✅ |
| M & J & M.ver2 | 10 | 1510 | 2P | 106 | 1.00 | 1.00 | 0.07 | ✅ |
| Azat Akhtyamov | 11 | 1484 | 2P | 120 | 1.00 | 1.00 | 0.20 | ✅ |
| Azat Akhtyamov | 11 | 1484 | 4P | 208 | 1.00 | 1.00 | 0.10 | ✅ |
| Azat Akhtyamov | 11 | 1484 | 2P | 499 | 1.00 | 1.00 | 0.06 | ✅ |
| Azat Akhtyamov | 11 | 1484 | 2P | 211 | 1.00 | 1.00 | 0.04 | ✅ |
| Azat Akhtyamov | 11 | 1484 | 4P | 223 | 1.00 | 1.00 | 0.02 | ✅ |
| Azat Akhtyamov | 11 | 1484 | 2P | 202 | 1.00 | 1.00 | 0.08 | ✅ |
| Azat Akhtyamov | 11 | 1484 | 4P | 165 | 1.00 | 1.00 | 0.03 | ✅ |
| Audun Ljone Henriksen | 12 | 1479 | 2P | 139 | 0.96 | 1.00 | 0.13 | ✅ |
| Audun Ljone Henriksen | 12 | 1479 | 4P | 321 | 0.96 | 1.00 | 0.04 | ✅ |
| Audun Ljone Henriksen | 12 | 1479 | 2P | 248 | 0.97 | 1.00 | 0.12 | ✅ |
| Audun Ljone Henriksen | 12 | 1479 | 4P | 253 | 0.96 | 1.00 | 0.03 | ✅ |
| dragon warrior | 13 | 1476 | 4P | 164 | 1.00 | 1.00 | 0.06 | ✅ |
| dragon warrior | 13 | 1476 | 4P | 178 | 1.00 | 1.00 | 0.02 | ✅ |
| One Man Wrecking Machine | 14 | 1457 | 4P | 321 | 1.00 | 1.00 | 0.09 | ✅ |
| One Man Wrecking Machine | 14 | 1457 | 2P | 162 | 1.00 | 1.00 | 0.11 | ✅ |
| One Man Wrecking Machine | 14 | 1457 | 2P | 127 | 1.00 | 1.00 | 0.08 | ✅ |
| One Man Wrecking Machine | 14 | 1457 | 4P | 499 | 1.00 | 1.00 | 0.01 | ✅ |
| One Man Wrecking Machine | 14 | 1457 | 4P | 499 | 1.00 | 1.00 | 0.08 | ✅ |
| One Man Wrecking Machine | 14 | 1457 | 4P | 253 | 1.00 | 1.00 | 0.14 | ✅ |
| 213tubo | 15 | 1455 | 4P | 109 | 1.00 | 1.00 | 0.07 | ✅ |
| 213tubo | 15 | 1455 | 4P | 193 | 1.00 | 1.00 | 0.11 | ✅ |
| 213tubo | 15 | 1455 | 2P | 162 | 1.00 | 1.00 | 0.05 | ✅ |
| 213tubo | 15 | 1455 | 4P | 165 | 1.00 | 1.00 | 0.09 | ✅ |
| 213tubo | 15 | 1455 | 2P | 183 | 1.00 | 1.00 | 0.04 | ✅ |

## Selection divergence (clone vs baseline, same obs)

`pair_prec` = of the clone's launches, fraction producer also makes (low = clone makes launches producer wouldn't = different/extra targets). `pair_rec` = of producer's launches, fraction the clone also makes (low = clone SKIPS launches producer makes). `src_*` = same at source-planet granularity. `tgt_match|src` = on shared source planets, fraction with the SAME target. 1.0 = identical to producer; lower = more divergent.

| vs | metric | state class | n | mean | sd |
|---|---|---|---|---|---|
| prod | pair_prec | all=all | 15773 | 0.178 | 0.326 |
| prod | pair_prec | contested=contested | 3598 | 0.165 | 0.330 |
| prod | pair_prec | contested=uncontested | 12175 | 0.182 | 0.325 |
| prod | pair_prec | fmt=2P | 10792 | 0.178 | 0.321 |
| prod | pair_prec | fmt=2P|contested=uncontested | 10792 | 0.178 | 0.321 |
| prod | pair_prec | fmt=2P|phase=end | 1888 | 0.141 | 0.278 |
| prod | pair_prec | fmt=2P|phase=mid | 6940 | 0.170 | 0.307 |
| prod | pair_prec | fmt=2P|phase=open | 1964 | 0.243 | 0.391 |
| prod | pair_prec | fmt=2P|standing=ahead | 4302 | 0.225 | 0.369 |
| prod | pair_prec | fmt=2P|standing=behind | 6490 | 0.147 | 0.280 |
| prod | pair_prec | fmt=4P | 4981 | 0.177 | 0.337 |
| prod | pair_prec | fmt=4P|contested=contested | 3598 | 0.165 | 0.330 |
| prod | pair_prec | fmt=4P|contested=uncontested | 1383 | 0.209 | 0.350 |
| prod | pair_prec | fmt=4P|phase=end | 497 | 0.199 | 0.354 |
| prod | pair_prec | fmt=4P|phase=mid | 3358 | 0.172 | 0.324 |
| prod | pair_prec | fmt=4P|phase=open | 1126 | 0.182 | 0.366 |
| prod | pair_prec | fmt=4P|standing=ahead | 1619 | 0.178 | 0.320 |
| prod | pair_prec | fmt=4P|standing=behind | 3362 | 0.176 | 0.344 |
| prod | pair_prec | phase=end | 2385 | 0.153 | 0.296 |
| prod | pair_prec | phase=mid | 10298 | 0.171 | 0.313 |
| prod | pair_prec | phase=open | 3090 | 0.221 | 0.383 |
| prod | pair_prec | standing=ahead | 5921 | 0.212 | 0.357 |
| prod | pair_prec | standing=behind | 9852 | 0.157 | 0.304 |
| prod | pair_rec | all=all | 20735 | 0.105 | 0.239 |
| prod | pair_rec | contested=contested | 3844 | 0.126 | 0.276 |
| prod | pair_rec | contested=uncontested | 16891 | 0.101 | 0.230 |
| prod | pair_rec | fmt=2P | 14884 | 0.100 | 0.229 |
| prod | pair_rec | fmt=2P|contested=uncontested | 14884 | 0.100 | 0.229 |
| prod | pair_rec | fmt=2P|phase=end | 3235 | 0.056 | 0.152 |
| prod | pair_rec | fmt=2P|phase=mid | 9572 | 0.095 | 0.214 |
| prod | pair_rec | fmt=2P|phase=open | 2077 | 0.189 | 0.339 |
| prod | pair_rec | fmt=2P|standing=ahead | 7942 | 0.084 | 0.223 |
| prod | pair_rec | fmt=2P|standing=behind | 6942 | 0.118 | 0.234 |
| prod | pair_rec | fmt=4P | 5851 | 0.119 | 0.264 |
| prod | pair_rec | fmt=4P|contested=contested | 3844 | 0.126 | 0.276 |
| prod | pair_rec | fmt=4P|contested=uncontested | 2007 | 0.105 | 0.237 |
| prod | pair_rec | fmt=4P|phase=end | 791 | 0.069 | 0.181 |
| prod | pair_rec | fmt=4P|phase=mid | 3948 | 0.119 | 0.255 |
| prod | pair_rec | fmt=4P|phase=open | 1112 | 0.155 | 0.328 |
| prod | pair_rec | fmt=4P|standing=ahead | 2164 | 0.123 | 0.268 |
| prod | pair_rec | fmt=4P|standing=behind | 3687 | 0.117 | 0.261 |
| prod | pair_rec | phase=end | 4026 | 0.058 | 0.159 |
| prod | pair_rec | phase=mid | 13520 | 0.102 | 0.227 |
| prod | pair_rec | phase=open | 3189 | 0.177 | 0.336 |
| prod | pair_rec | standing=ahead | 10106 | 0.093 | 0.234 |
| prod | pair_rec | standing=behind | 10629 | 0.117 | 0.243 |
| prod | src_prec | all=all | 15773 | 0.387 | 0.419 |
| prod | src_prec | contested=contested | 3598 | 0.369 | 0.436 |
| prod | src_prec | contested=uncontested | 12175 | 0.392 | 0.414 |
| prod | src_prec | fmt=2P | 10792 | 0.389 | 0.411 |
| prod | src_prec | fmt=2P|contested=uncontested | 10792 | 0.389 | 0.411 |
| prod | src_prec | fmt=2P|phase=end | 1888 | 0.362 | 0.394 |
| prod | src_prec | fmt=2P|phase=mid | 6940 | 0.386 | 0.402 |
| prod | src_prec | fmt=2P|phase=open | 1964 | 0.423 | 0.455 |
| prod | src_prec | fmt=2P|standing=ahead | 4302 | 0.492 | 0.446 |
| prod | src_prec | fmt=2P|standing=behind | 6490 | 0.320 | 0.371 |
| prod | src_prec | fmt=4P | 4981 | 0.383 | 0.435 |
| prod | src_prec | fmt=4P|contested=contested | 3598 | 0.369 | 0.436 |
| prod | src_prec | fmt=4P|contested=uncontested | 1383 | 0.418 | 0.432 |
| prod | src_prec | fmt=4P|phase=end | 497 | 0.382 | 0.444 |
| prod | src_prec | fmt=4P|phase=mid | 3358 | 0.393 | 0.427 |
| prod | src_prec | fmt=4P|phase=open | 1126 | 0.353 | 0.454 |
| prod | src_prec | fmt=4P|standing=ahead | 1619 | 0.366 | 0.403 |
| prod | src_prec | fmt=4P|standing=behind | 3362 | 0.391 | 0.450 |
| prod | src_prec | phase=end | 2385 | 0.366 | 0.405 |
| prod | src_prec | phase=mid | 10298 | 0.389 | 0.410 |
| prod | src_prec | phase=open | 3090 | 0.397 | 0.456 |
| prod | src_prec | standing=ahead | 5921 | 0.458 | 0.438 |
| prod | src_prec | standing=behind | 9852 | 0.344 | 0.401 |
| prod | src_rec | all=all | 20735 | 0.224 | 0.326 |
| prod | src_rec | contested=contested | 3844 | 0.268 | 0.361 |
| prod | src_rec | contested=uncontested | 16891 | 0.214 | 0.316 |
| prod | src_rec | fmt=2P | 14884 | 0.214 | 0.315 |
| prod | src_rec | fmt=2P|contested=uncontested | 14884 | 0.214 | 0.315 |
| prod | src_rec | fmt=2P|phase=end | 3235 | 0.143 | 0.254 |
| prod | src_rec | fmt=2P|phase=mid | 9572 | 0.216 | 0.309 |
| prod | src_rec | fmt=2P|phase=open | 2077 | 0.317 | 0.393 |
| prod | src_rec | fmt=2P|standing=ahead | 7942 | 0.173 | 0.301 |
| prod | src_rec | fmt=2P|standing=behind | 6942 | 0.261 | 0.324 |
| prod | src_rec | fmt=4P | 5851 | 0.248 | 0.349 |
| prod | src_rec | fmt=4P|contested=contested | 3844 | 0.268 | 0.361 |
| prod | src_rec | fmt=4P|contested=uncontested | 2007 | 0.209 | 0.322 |
| prod | src_rec | fmt=4P|phase=end | 791 | 0.134 | 0.256 |
| prod | src_rec | fmt=4P|phase=mid | 3948 | 0.261 | 0.348 |
| prod | src_rec | fmt=4P|phase=open | 1112 | 0.283 | 0.394 |
| prod | src_rec | fmt=4P|standing=ahead | 2164 | 0.236 | 0.341 |
| prod | src_rec | fmt=4P|standing=behind | 3687 | 0.255 | 0.354 |
| prod | src_rec | phase=end | 4026 | 0.141 | 0.254 |
| prod | src_rec | phase=mid | 13520 | 0.229 | 0.321 |
| prod | src_rec | phase=open | 3189 | 0.305 | 0.394 |
| prod | src_rec | standing=ahead | 10106 | 0.187 | 0.311 |
| prod | src_rec | standing=behind | 10629 | 0.259 | 0.335 |
| prod | tgt_match|src | all=all | 8733 | 0.451 | 0.459 |
| prod | tgt_match|src | contested=contested | 1685 | 0.450 | 0.473 |
| prod | tgt_match|src | contested=uncontested | 7048 | 0.451 | 0.456 |
| prod | tgt_match|src | fmt=2P | 6268 | 0.445 | 0.455 |
| prod | tgt_match|src | fmt=2P|contested=uncontested | 6268 | 0.445 | 0.455 |
| prod | tgt_match|src | fmt=2P|phase=end | 1121 | 0.388 | 0.439 |
| prod | tgt_match|src | fmt=2P|phase=mid | 4157 | 0.432 | 0.451 |
| prod | tgt_match|src | fmt=2P|phase=open | 990 | 0.569 | 0.468 |
| prod | tgt_match|src | fmt=2P|standing=ahead | 2711 | 0.451 | 0.464 |
| prod | tgt_match|src | fmt=2P|standing=behind | 3557 | 0.442 | 0.448 |
| prod | tgt_match|src | fmt=4P | 2465 | 0.464 | 0.470 |
| prod | tgt_match|src | fmt=4P|contested=contested | 1685 | 0.450 | 0.473 |
| prod | tgt_match|src | fmt=4P|contested=uncontested | 780 | 0.493 | 0.461 |
| prod | tgt_match|src | fmt=4P|phase=end | 231 | 0.515 | 0.458 |
| prod | tgt_match|src | fmt=4P|phase=mid | 1784 | 0.445 | 0.465 |
| prod | tgt_match|src | fmt=4P|phase=open | 450 | 0.512 | 0.487 |
| prod | tgt_match|src | fmt=4P|standing=ahead | 894 | 0.479 | 0.463 |
| prod | tgt_match|src | fmt=4P|standing=behind | 1571 | 0.455 | 0.473 |
| prod | tgt_match|src | phase=end | 1352 | 0.410 | 0.445 |
| prod | tgt_match|src | phase=mid | 5941 | 0.436 | 0.455 |
| prod | tgt_match|src | phase=open | 1440 | 0.551 | 0.475 |
| prod | tgt_match|src | standing=ahead | 3605 | 0.458 | 0.464 |
| prod | tgt_match|src | standing=behind | 5128 | 0.446 | 0.456 |
| v5 | pair_prec | all=all | 15773 | 0.171 | 0.322 |
| v5 | pair_prec | contested=contested | 3598 | 0.159 | 0.325 |
| v5 | pair_prec | contested=uncontested | 12175 | 0.174 | 0.321 |
| v5 | pair_prec | fmt=2P | 10792 | 0.173 | 0.319 |
| v5 | pair_prec | fmt=2P|contested=uncontested | 10792 | 0.173 | 0.319 |
| v5 | pair_prec | fmt=2P|phase=end | 1888 | 0.127 | 0.271 |
| v5 | pair_prec | fmt=2P|phase=mid | 6940 | 0.167 | 0.307 |
| v5 | pair_prec | fmt=2P|phase=open | 1964 | 0.236 | 0.388 |
| v5 | pair_prec | fmt=2P|standing=ahead | 4302 | 0.222 | 0.368 |
| v5 | pair_prec | fmt=2P|standing=behind | 6490 | 0.140 | 0.277 |
| v5 | pair_prec | fmt=4P | 4981 | 0.166 | 0.328 |
| v5 | pair_prec | fmt=4P|contested=contested | 3598 | 0.159 | 0.325 |
| v5 | pair_prec | fmt=4P|contested=uncontested | 1383 | 0.185 | 0.333 |
| v5 | pair_prec | fmt=4P|phase=end | 497 | 0.169 | 0.328 |
| v5 | pair_prec | fmt=4P|phase=mid | 3358 | 0.162 | 0.316 |
| v5 | pair_prec | fmt=4P|phase=open | 1126 | 0.176 | 0.360 |
| v5 | pair_prec | fmt=4P|standing=ahead | 1619 | 0.174 | 0.318 |
| v5 | pair_prec | fmt=4P|standing=behind | 3362 | 0.162 | 0.332 |
| v5 | pair_prec | phase=end | 2385 | 0.135 | 0.284 |
| v5 | pair_prec | phase=mid | 10298 | 0.166 | 0.310 |
| v5 | pair_prec | phase=open | 3090 | 0.214 | 0.379 |
| v5 | pair_prec | standing=ahead | 5921 | 0.209 | 0.356 |
| v5 | pair_prec | standing=behind | 9852 | 0.148 | 0.297 |
| v5 | pair_rec | all=all | 19832 | 0.117 | 0.258 |
| v5 | pair_rec | contested=contested | 3598 | 0.135 | 0.288 |
| v5 | pair_rec | contested=uncontested | 16234 | 0.113 | 0.251 |
| v5 | pair_rec | fmt=2P | 14279 | 0.113 | 0.251 |
| v5 | pair_rec | fmt=2P|contested=uncontested | 14279 | 0.113 | 0.251 |
| v5 | pair_rec | fmt=2P|phase=end | 3018 | 0.066 | 0.185 |
| v5 | pair_rec | fmt=2P|phase=mid | 9255 | 0.110 | 0.239 |
| v5 | pair_rec | fmt=2P|phase=open | 2006 | 0.200 | 0.352 |
| v5 | pair_rec | fmt=2P|standing=ahead | 7772 | 0.093 | 0.237 |
| v5 | pair_rec | fmt=2P|standing=behind | 6507 | 0.137 | 0.265 |
| v5 | pair_rec | fmt=4P | 5553 | 0.126 | 0.275 |
| v5 | pair_rec | fmt=4P|contested=contested | 3598 | 0.135 | 0.288 |
| v5 | pair_rec | fmt=4P|contested=uncontested | 1955 | 0.108 | 0.250 |
| v5 | pair_rec | fmt=4P|phase=end | 743 | 0.080 | 0.214 |
| v5 | pair_rec | fmt=4P|phase=mid | 3767 | 0.124 | 0.266 |
| v5 | pair_rec | fmt=4P|phase=open | 1043 | 0.164 | 0.336 |
| v5 | pair_rec | fmt=4P|standing=ahead | 2112 | 0.128 | 0.276 |
| v5 | pair_rec | fmt=4P|standing=behind | 3441 | 0.124 | 0.275 |
| v5 | pair_rec | phase=end | 3761 | 0.069 | 0.191 |
| v5 | pair_rec | phase=mid | 13022 | 0.114 | 0.247 |
| v5 | pair_rec | phase=open | 3049 | 0.188 | 0.347 |
| v5 | pair_rec | standing=ahead | 9884 | 0.101 | 0.247 |
| v5 | pair_rec | standing=behind | 9948 | 0.132 | 0.268 |
| v5 | src_prec | all=all | 15773 | 0.338 | 0.408 |
| v5 | src_prec | contested=contested | 3598 | 0.336 | 0.426 |
| v5 | src_prec | contested=uncontested | 12175 | 0.338 | 0.402 |
| v5 | src_prec | fmt=2P | 10792 | 0.334 | 0.399 |
| v5 | src_prec | fmt=2P|contested=uncontested | 10792 | 0.334 | 0.399 |
| v5 | src_prec | fmt=2P|phase=end | 1888 | 0.270 | 0.371 |
| v5 | src_prec | fmt=2P|phase=mid | 6940 | 0.335 | 0.389 |
| v5 | src_prec | fmt=2P|phase=open | 1964 | 0.391 | 0.448 |
| v5 | src_prec | fmt=2P|standing=ahead | 4302 | 0.439 | 0.443 |
| v5 | src_prec | fmt=2P|standing=behind | 6490 | 0.264 | 0.351 |
| v5 | src_prec | fmt=4P | 4981 | 0.346 | 0.425 |
| v5 | src_prec | fmt=4P|contested=contested | 3598 | 0.336 | 0.426 |
| v5 | src_prec | fmt=4P|contested=uncontested | 1383 | 0.372 | 0.421 |
| v5 | src_prec | fmt=4P|phase=end | 497 | 0.313 | 0.420 |
| v5 | src_prec | fmt=4P|phase=mid | 3358 | 0.358 | 0.418 |
| v5 | src_prec | fmt=4P|phase=open | 1126 | 0.326 | 0.444 |
| v5 | src_prec | fmt=4P|standing=ahead | 1619 | 0.343 | 0.398 |
| v5 | src_prec | fmt=4P|standing=behind | 3362 | 0.347 | 0.437 |
| v5 | src_prec | phase=end | 2385 | 0.279 | 0.382 |
| v5 | src_prec | phase=mid | 10298 | 0.342 | 0.399 |
| v5 | src_prec | phase=open | 3090 | 0.367 | 0.448 |
| v5 | src_prec | standing=ahead | 5921 | 0.413 | 0.433 |
| v5 | src_prec | standing=behind | 9852 | 0.292 | 0.384 |
| v5 | src_rec | all=all | 19832 | 0.228 | 0.339 |
| v5 | src_rec | contested=contested | 3598 | 0.273 | 0.368 |
| v5 | src_rec | contested=uncontested | 16234 | 0.218 | 0.331 |
| v5 | src_rec | fmt=2P | 14279 | 0.219 | 0.331 |
| v5 | src_rec | fmt=2P|contested=uncontested | 14279 | 0.219 | 0.331 |
| v5 | src_rec | fmt=2P|phase=end | 3018 | 0.138 | 0.268 |
| v5 | src_rec | fmt=2P|phase=mid | 9255 | 0.223 | 0.325 |
| v5 | src_rec | fmt=2P|phase=open | 2006 | 0.322 | 0.403 |
| v5 | src_rec | fmt=2P|standing=ahead | 7772 | 0.179 | 0.314 |
| v5 | src_rec | fmt=2P|standing=behind | 6507 | 0.266 | 0.344 |
| v5 | src_rec | fmt=4P | 5553 | 0.251 | 0.357 |
| v5 | src_rec | fmt=4P|contested=contested | 3598 | 0.273 | 0.368 |
| v5 | src_rec | fmt=4P|contested=uncontested | 1955 | 0.212 | 0.334 |
| v5 | src_rec | fmt=4P|phase=end | 743 | 0.145 | 0.286 |
| v5 | src_rec | fmt=4P|phase=mid | 3767 | 0.263 | 0.354 |
| v5 | src_rec | fmt=4P|phase=open | 1043 | 0.287 | 0.398 |
| v5 | src_rec | fmt=4P|standing=ahead | 2112 | 0.240 | 0.349 |
| v5 | src_rec | fmt=4P|standing=behind | 3441 | 0.258 | 0.362 |
| v5 | src_rec | phase=end | 3761 | 0.139 | 0.271 |
| v5 | src_rec | phase=mid | 13022 | 0.234 | 0.335 |
| v5 | src_rec | phase=open | 3049 | 0.310 | 0.402 |
| v5 | src_rec | standing=ahead | 9884 | 0.192 | 0.323 |
| v5 | src_rec | standing=behind | 9948 | 0.264 | 0.351 |
| v5 | tgt_match|src | all=all | 7800 | 0.497 | 0.468 |
| v5 | tgt_match|src | contested=contested | 1554 | 0.477 | 0.476 |
| v5 | tgt_match|src | contested=uncontested | 6246 | 0.502 | 0.466 |
| v5 | tgt_match|src | fmt=2P | 5525 | 0.503 | 0.465 |
| v5 | tgt_match|src | fmt=2P|contested=uncontested | 5525 | 0.503 | 0.465 |
| v5 | tgt_match|src | fmt=2P|phase=end | 850 | 0.470 | 0.460 |
| v5 | tgt_match|src | fmt=2P|phase=mid | 3747 | 0.487 | 0.463 |
| v5 | tgt_match|src | fmt=2P|phase=open | 928 | 0.599 | 0.466 |
| v5 | tgt_match|src | fmt=2P|standing=ahead | 2480 | 0.495 | 0.469 |
| v5 | tgt_match|src | fmt=2P|standing=behind | 3045 | 0.510 | 0.462 |
| v5 | tgt_match|src | fmt=4P | 2275 | 0.481 | 0.474 |
| v5 | tgt_match|src | fmt=4P|contested=contested | 1554 | 0.477 | 0.476 |
| v5 | tgt_match|src | fmt=4P|contested=uncontested | 721 | 0.491 | 0.469 |
| v5 | tgt_match|src | fmt=4P|phase=end | 198 | 0.531 | 0.470 |
| v5 | tgt_match|src | fmt=4P|phase=mid | 1658 | 0.461 | 0.469 |
| v5 | tgt_match|src | fmt=4P|phase=open | 419 | 0.535 | 0.487 |
| v5 | tgt_match|src | fmt=4P|standing=ahead | 847 | 0.500 | 0.467 |
| v5 | tgt_match|src | fmt=4P|standing=behind | 1428 | 0.470 | 0.477 |
| v5 | tgt_match|src | phase=end | 1048 | 0.481 | 0.462 |
| v5 | tgt_match|src | phase=mid | 5405 | 0.479 | 0.465 |
| v5 | tgt_match|src | phase=open | 1347 | 0.579 | 0.474 |
| v5 | tgt_match|src | standing=ahead | 3327 | 0.496 | 0.468 |
| v5 | tgt_match|src | standing=behind | 4473 | 0.497 | 0.468 |

## Ranked behavioural divergences (clone − baseline)

Positive `Δ` = clone does MORE of the axis than the baseline on the SAME obs. `base` = baseline's own mean (for scale); `pct` = Δ as % of base; `t` = t-stat (|t|>=3 kept); `eff` = |Δ|/sd; `cons` = sign-agreement; `nz` = fraction of turns exercising the axis. Ranked by |t|·eff.

| # | vs | axis | state class | n | Δ | base | pct | t | eff | cons | nz |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | prod | active | fmt=2P|standing=ahead | 9945 | -0.366 | 0.80 | -46% | -60.7 | 0.61 | 0.43 | 0.50 |
| 2 | prod | active | fmt=2P|phase=end | 3278 | -0.411 | 0.99 | -42% | -45.6 | 0.80 | 0.42 | 0.44 |
| 3 | v5 | active | fmt=2P|standing=ahead | 9945 | -0.349 | 0.78 | -45% | -56.6 | 0.57 | 0.42 | 0.50 |
| 4 | prod | active | standing=ahead | 13213 | -0.317 | 0.76 | -41% | -60.0 | 0.52 | 0.39 | 0.47 |
| 5 | v5 | active | standing=ahead | 13213 | -0.300 | 0.75 | -40% | -55.7 | 0.48 | 0.39 | 0.47 |
| 6 | prod | ships | fmt=2P|standing=behind | 8283 | -145.638 | 198.86 | -73% | -50.5 | 0.55 | 0.67 | 0.89 |
| 7 | prod | ships | standing=behind | 16364 | -89.192 | 130.30 | -68% | -55.3 | 0.43 | 0.52 | 0.73 |
| 8 | prod | waves | fmt=2P|standing=ahead | 9945 | -2.131 | 3.26 | -65% | -49.3 | 0.49 | 0.65 | 0.79 |
| 9 | prod | sources | fmt=2P|standing=ahead | 9945 | -2.125 | 3.25 | -65% | -49.3 | 0.49 | 0.65 | 0.79 |
| 10 | prod | active | contested=uncontested | 21266 | -0.222 | 0.79 | -28% | -57.3 | 0.39 | 0.29 | 0.37 |
| 11 | prod | active | phase=end | 4815 | -0.341 | 0.84 | -41% | -41.9 | 0.60 | 0.39 | 0.43 |
| 12 | prod | active | fmt=2P | 18228 | -0.224 | 0.82 | -27% | -54.3 | 0.40 | 0.29 | 0.36 |
| 13 | prod | active | fmt=2P|contested=uncontested | 18228 | -0.224 | 0.82 | -27% | -54.3 | 0.40 | 0.29 | 0.36 |
| 14 | prod | sources | standing=ahead | 13213 | -1.800 | 2.98 | -60% | -50.3 | 0.44 | 0.60 | 0.76 |
| 15 | prod | waves | standing=ahead | 13213 | -1.803 | 2.98 | -60% | -50.2 | 0.44 | 0.60 | 0.76 |
| 16 | prod | active | fmt=2P|phase=mid | 11050 | -0.238 | 0.87 | -27% | -45.3 | 0.43 | 0.30 | 0.36 |
| 17 | prod | waves | phase=end | 4815 | -2.220 | 3.77 | -59% | -38.1 | 0.55 | 0.65 | 0.82 |
| 18 | prod | sources | phase=end | 4815 | -2.217 | 3.76 | -59% | -38.1 | 0.55 | 0.65 | 0.82 |
| 19 | prod | ships | standing=ahead | 13213 | -732.783 | 799.70 | -92% | -45.1 | 0.39 | 0.64 | 0.81 |
| 20 | prod | ships | contested=uncontested | 21266 | -508.398 | 569.42 | -89% | -49.2 | 0.34 | 0.66 | 0.84 |
| 21 | prod | waves | fmt=2P|phase=end | 3278 | -2.657 | 4.70 | -56% | -33.2 | 0.58 | 0.75 | 0.93 |
| 22 | prod | sources | fmt=2P|phase=end | 3278 | -2.646 | 4.69 | -56% | -33.1 | 0.58 | 0.75 | 0.93 |
| 23 | v5 | ships | fmt=2P|standing=behind | 8283 | -97.648 | 150.87 | -65% | -40.0 | 0.44 | 0.58 | 0.88 |
| 24 | prod | active | all=all | 29577 | -0.168 | 0.70 | -24% | -50.7 | 0.29 | 0.26 | 0.35 |
| 25 | v5 | active | fmt=2P|phase=end | 3278 | -0.345 | 0.92 | -37% | -32.6 | 0.57 | 0.41 | 0.49 |
| 26 | v5 | ships | standing=ahead | 13213 | -641.878 | 708.79 | -91% | -43.2 | 0.38 | 0.61 | 0.81 |
| 27 | prod | ships | fmt=2P|standing=ahead | 9945 | -831.407 | 892.94 | -93% | -40.7 | 0.41 | 0.68 | 0.84 |
| 28 | v5 | active | contested=uncontested | 21266 | -0.191 | 0.76 | -25% | -47.2 | 0.32 | 0.29 | 0.38 |
| 29 | prod | waves | contested=uncontested | 21266 | -1.224 | 2.85 | -43% | -47.1 | 0.32 | 0.55 | 0.76 |
| 30 | prod | sources | contested=uncontested | 21266 | -1.220 | 2.84 | -43% | -47.0 | 0.32 | 0.55 | 0.76 |
| 31 | prod | ships | all=all | 29577 | -376.705 | 429.34 | -88% | -49.9 | 0.29 | 0.58 | 0.77 |
| 32 | prod | waves | all=all | 29577 | -0.962 | 2.33 | -41% | -49.9 | 0.29 | 0.48 | 0.69 |
| 33 | v5 | ships | standing=behind | 16364 | -60.542 | 101.65 | -60% | -44.5 | 0.35 | 0.46 | 0.72 |
| 34 | prod | sources | all=all | 29577 | -0.960 | 2.32 | -41% | -49.8 | 0.29 | 0.48 | 0.69 |
| 35 | prod | ships | fmt=2P | 18228 | -519.786 | 577.54 | -90% | -45.2 | 0.33 | 0.68 | 0.86 |
| 36 | prod | ships | fmt=2P|contested=uncontested | 18228 | -519.786 | 577.54 | -90% | -45.2 | 0.33 | 0.68 | 0.86 |
| 37 | v5 | ships | contested=uncontested | 21266 | -433.220 | 494.24 | -88% | -45.8 | 0.31 | 0.60 | 0.83 |
| 38 | prod | ships | phase=end | 4815 | -1229.861 | 1288.49 | -95% | -33.9 | 0.49 | 0.76 | 0.88 |
| 39 | prod | waves | fmt=2P | 18228 | -1.264 | 2.97 | -43% | -44.2 | 0.33 | 0.57 | 0.78 |
| 40 | prod | waves | fmt=2P|contested=uncontested | 18228 | -1.264 | 2.97 | -43% | -44.2 | 0.33 | 0.57 | 0.78 |
| 41 | v5 | active | fmt=2P | 18228 | -0.191 | 0.78 | -24% | -44.1 | 0.33 | 0.29 | 0.38 |
| 42 | v5 | active | fmt=2P|contested=uncontested | 18228 | -0.191 | 0.78 | -24% | -44.1 | 0.33 | 0.29 | 0.38 |
| 43 | prod | sources | fmt=2P | 18228 | -1.257 | 2.96 | -42% | -44.0 | 0.33 | 0.57 | 0.78 |
| 44 | prod | sources | fmt=2P|contested=uncontested | 18228 | -1.257 | 2.96 | -42% | -44.0 | 0.33 | 0.57 | 0.78 |
| 45 | prod | ships | fmt=2P|phase=end | 3278 | -1518.988 | 1589.02 | -96% | -30.8 | 0.54 | 0.90 | 1.00 |
| 46 | v5 | ships | fmt=2P|standing=ahead | 9945 | -715.235 | 776.77 | -92% | -38.6 | 0.39 | 0.65 | 0.83 |
| 47 | prod | active | phase=mid | 18402 | -0.175 | 0.73 | -24% | -42.8 | 0.32 | 0.26 | 0.34 |
| 48 | v5 | waves | fmt=2P|standing=ahead | 9945 | -1.516 | 2.65 | -57% | -37.8 | 0.38 | 0.61 | 0.77 |
| 49 | v5 | sources | fmt=2P|standing=ahead | 9945 | -1.515 | 2.64 | -57% | -37.8 | 0.38 | 0.61 | 0.77 |
| 50 | v5 | ships | all=all | 29577 | -320.244 | 372.88 | -86% | -46.5 | 0.27 | 0.53 | 0.76 |
| 51 | prod | ships | fmt=2P|phase=mid | 11050 | -402.672 | 468.59 | -86% | -38.4 | 0.37 | 0.72 | 0.91 |
| 52 | v5 | active | fmt=2P|phase=mid | 11050 | -0.210 | 0.84 | -25% | -38.4 | 0.37 | 0.29 | 0.37 |
| 53 | prod | ships | phase=mid | 18402 | -279.629 | 341.15 | -82% | -42.2 | 0.31 | 0.60 | 0.80 |
| 54 | v5 | ships | fmt=2P | 18228 | -434.597 | 492.35 | -88% | -41.7 | 0.31 | 0.62 | 0.85 |
| 55 | v5 | ships | fmt=2P|contested=uncontested | 18228 | -434.597 | 492.35 | -88% | -41.7 | 0.31 | 0.62 | 0.85 |
| 56 | v5 | active | phase=end | 4815 | -0.286 | 0.78 | -37% | -32.1 | 0.46 | 0.38 | 0.46 |
| 57 | v5 | sources | standing=ahead | 13213 | -1.298 | 2.47 | -52% | -38.9 | 0.34 | 0.56 | 0.74 |
| 58 | v5 | waves | standing=ahead | 13213 | -1.297 | 2.48 | -52% | -38.8 | 0.34 | 0.56 | 0.74 |
| 59 | v5 | ships | phase=end | 4815 | -1012.686 | 1071.31 | -95% | -31.1 | 0.45 | 0.69 | 0.86 |
| 60 | v5 | ships | fmt=2P|phase=end | 3278 | -1212.739 | 1282.77 | -95% | -27.7 | 0.48 | 0.81 | 0.99 |

## Target-disagreement profile (clone target − producer target)

_Plan 1, the confound-free residual: on the **shared source planets** where the clone and bare `producer` BOTH launch but resolve to **different targets**, what is systematically different about the target the clone picks vs the one producer's flow-diff argmax picks?_

- Shared-source decisions: **11896**; of which the clone aimed elsewhere than producer: **6084** (**51%** disagreement rate — matches `tgt_match|src`).

Each row = a target-feature axis conditioned on a state class. `Δ` = mean(clone_target_feature − producer_target_feature) over disagreements (positive = clone's chosen target scores HIGHER on the axis). `base` = producer-target mean (scale); `pct` = Δ as % of base; `t` = t-stat (|t|>=3 kept); `eff` = |Δ|/sd; `cons` = sign-agreement; `nz` = fraction of disagreements where the axis differs. Axes: `dist` (source→target euclidean), `prod`, `garrison`, `contested` (>=2 owners inbound), `orbiting`, and owner-class indicators `is_enemy`/`is_neutral`/`is_own`. Ranked by |t|·eff.

| # | axis | state class | n | Δ | base | pct | t | eff | cons | nz |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | prod | fmt=2P|standing=behind | 2738 | +0.469 | 2.47 | +19% | +14.0 | 0.27 | 0.48 | 0.78 |
| 2 | prod | fmt=2P | 4530 | +0.397 | 2.61 | +15% | +14.8 | 0.22 | 0.46 | 0.76 |
| 3 | prod | fmt=2P|contested=uncontested | 4530 | +0.397 | 2.61 | +15% | +14.8 | 0.22 | 0.46 | 0.76 |
| 4 | prod | fmt=2P|phase=mid | 3089 | +0.425 | 2.60 | +16% | +13.1 | 0.24 | 0.47 | 0.77 |
| 5 | dist | standing=ahead | 2377 | -6.259 | 32.37 | -19% | -11.8 | 0.24 | 0.64 | 1.00 |
| 6 | dist | contested=uncontested | 5048 | -4.699 | 30.14 | -16% | -13.3 | 0.19 | 0.63 | 1.00 |
| 7 | dist | fmt=2P | 4530 | -4.913 | 30.63 | -16% | -13.0 | 0.19 | 0.63 | 1.00 |
| 8 | dist | fmt=2P|contested=uncontested | 4530 | -4.913 | 30.63 | -16% | -13.0 | 0.19 | 0.63 | 1.00 |
| 9 | prod | all=all | 6084 | +0.323 | 2.72 | +12% | +13.5 | 0.17 | 0.44 | 0.77 |
| 10 | dist | fmt=2P|standing=ahead | 1792 | -6.725 | 33.49 | -20% | -10.8 | 0.26 | 0.65 | 1.00 |
| 11 | prod | phase=mid | 4275 | +0.355 | 2.69 | +13% | +12.5 | 0.19 | 0.45 | 0.78 |
| 12 | dist | all=all | 6084 | -4.265 | 29.71 | -14% | -13.1 | 0.17 | 0.62 | 1.00 |
| 13 | prod | contested=uncontested | 5048 | +0.325 | 2.70 | +12% | +12.6 | 0.18 | 0.44 | 0.77 |
| 14 | prod | standing=behind | 3707 | +0.344 | 2.63 | +13% | +11.4 | 0.19 | 0.46 | 0.78 |
| 15 | orbiting | fmt=2P|standing=behind | 2738 | -0.145 | 0.64 | -23% | -10.7 | 0.20 | 0.33 | 0.52 |
| 16 | garrison | phase=mid | 4275 | +19.313 | 30.95 | +62% | +11.3 | 0.17 | 0.58 | 0.97 |
| 17 | dist | fmt=2P|phase=mid | 3089 | -4.803 | 30.86 | -16% | -10.4 | 0.19 | 0.63 | 1.00 |
| 18 | dist | phase=open | 731 | -6.810 | 29.85 | -23% | -8.0 | 0.29 | 0.64 | 1.00 |
| 19 | is_own | fmt=2P|standing=ahead | 1792 | +0.142 | 0.39 | +36% | +9.2 | 0.22 | 0.29 | 0.45 |
| 20 | dist | fmt=2P|phase=open | 508 | -7.395 | 30.84 | -24% | -7.3 | 0.32 | 0.66 | 1.00 |
| 21 | garrison | fmt=4P|phase=mid | 1186 | +25.260 | 32.10 | +79% | +8.4 | 0.24 | 0.59 | 0.98 |
| 22 | garrison | fmt=4P | 1554 | +26.688 | 30.75 | +87% | +8.6 | 0.22 | 0.60 | 0.98 |
| 23 | dist | phase=mid | 4275 | -3.941 | 29.80 | -13% | -10.0 | 0.15 | 0.61 | 1.00 |
| 24 | is_own | standing=ahead | 2377 | +0.120 | 0.43 | +28% | +8.9 | 0.18 | 0.28 | 0.45 |
| 25 | garrison | contested=contested | 1036 | +21.236 | 31.30 | +68% | +7.7 | 0.24 | 0.61 | 0.98 |
| 26 | garrison | fmt=4P|contested=contested | 1036 | +21.236 | 31.30 | +68% | +7.7 | 0.24 | 0.61 | 0.98 |
| 27 | garrison | all=all | 6084 | +24.388 | 32.17 | +76% | +10.0 | 0.13 | 0.59 | 0.97 |
| 28 | contested | all=all | 6084 | +0.048 | 0.07 | +73% | +9.6 | 0.12 | 0.10 | 0.16 |
| 29 | contested | fmt=2P | 4530 | +0.052 | 0.06 | +86% | +9.1 | 0.14 | 0.10 | 0.15 |
| 30 | contested | fmt=2P|contested=uncontested | 4530 | +0.052 | 0.06 | +86% | +9.1 | 0.14 | 0.10 | 0.15 |
| 31 | garrison | phase=open | 731 | +9.677 | 20.03 | +48% | +6.6 | 0.24 | 0.60 | 0.97 |
| 32 | garrison | fmt=2P|phase=mid | 3089 | +17.029 | 30.51 | +56% | +8.3 | 0.15 | 0.58 | 0.97 |
| 33 | garrison | standing=behind | 3707 | +29.509 | 29.48 | +100% | +8.5 | 0.14 | 0.59 | 0.98 |
| 34 | is_enemy | fmt=4P|standing=behind | 969 | -0.142 | 0.54 | -26% | -6.8 | 0.22 | 0.29 | 0.44 |
| 35 | garrison | contested=uncontested | 5048 | +25.034 | 32.35 | +77% | +8.7 | 0.12 | 0.58 | 0.97 |
| 36 | dist | fmt=2P|standing=behind | 2738 | -3.727 | 28.75 | -13% | -7.9 | 0.15 | 0.62 | 1.00 |
| 37 | garrison | fmt=4P|standing=behind | 969 | +27.033 | 27.23 | +99% | +6.6 | 0.21 | 0.61 | 0.98 |
| 38 | is_enemy | fmt=4P | 1554 | -0.118 | 0.51 | -23% | -7.1 | 0.18 | 0.28 | 0.44 |
| 39 | prod | fmt=2P|phase=end | 933 | +0.403 | 2.48 | +16% | +6.5 | 0.21 | 0.46 | 0.78 |
| 40 | contested | contested=uncontested | 5048 | +0.044 | 0.06 | +73% | +8.3 | 0.12 | 0.09 | 0.15 |
| 41 | prod | standing=ahead | 2377 | +0.291 | 2.86 | +10% | +7.4 | 0.15 | 0.42 | 0.75 |
| 42 | garrison | fmt=4P|standing=ahead | 585 | +26.116 | 36.59 | +71% | +5.7 | 0.24 | 0.58 | 0.98 |
| 43 | contested | phase=mid | 4275 | +0.049 | 0.08 | +63% | +7.7 | 0.12 | 0.11 | 0.18 |
| 44 | contested | phase=end | 1078 | +0.054 | 0.02 | +290% | +6.2 | 0.19 | 0.07 | 0.09 |
| 45 | garrison | fmt=2P | 4530 | +23.599 | 32.66 | +72% | +7.6 | 0.11 | 0.58 | 0.97 |
| 46 | garrison | fmt=2P|contested=uncontested | 4530 | +23.599 | 32.66 | +72% | +7.6 | 0.11 | 0.58 | 0.97 |
| 47 | dist | standing=behind | 3707 | -2.987 | 28.00 | -11% | -7.3 | 0.12 | 0.61 | 1.00 |
| 48 | contested | standing=behind | 3707 | +0.045 | 0.06 | +75% | +7.2 | 0.12 | 0.10 | 0.15 |
| 49 | contested | fmt=2P|phase=end | 933 | +0.058 | 0.02 | +270% | +5.9 | 0.19 | 0.08 | 0.09 |
| 50 | contested | fmt=2P|phase=mid | 3089 | +0.051 | 0.07 | +71% | +7.0 | 0.13 | 0.11 | 0.17 |
| 51 | is_enemy | fmt=4P|phase=open | 223 | -0.197 | 0.43 | -46% | -4.5 | 0.30 | 0.33 | 0.47 |
| 52 | garrison | fmt=4P|phase=open | 223 | +13.637 | 22.08 | +62% | +4.5 | 0.30 | 0.65 | 0.99 |
| 53 | prod | fmt=2P|standing=ahead | 1792 | +0.287 | 2.82 | +10% | +6.4 | 0.15 | 0.41 | 0.74 |
| 54 | garrison | fmt=2P|standing=behind | 2738 | +30.385 | 30.27 | +100% | +6.8 | 0.13 | 0.58 | 0.97 |
| 55 | orbiting | standing=behind | 3707 | -0.082 | 0.61 | -14% | -7.1 | 0.12 | 0.29 | 0.51 |
| 56 | contested | fmt=2P|standing=behind | 2738 | +0.048 | 0.06 | +85% | +6.7 | 0.13 | 0.10 | 0.14 |
| 57 | is_enemy | standing=ahead | 2377 | -0.084 | 0.47 | -18% | -6.5 | 0.13 | 0.24 | 0.40 |
| 58 | is_neutral | fmt=2P|standing=ahead | 1792 | -0.055 | 0.13 | -43% | -6.2 | 0.15 | 0.10 | 0.14 |
| 59 | contested | fmt=2P|standing=ahead | 1792 | +0.057 | 0.06 | +89% | +6.2 | 0.15 | 0.11 | 0.16 |
| 60 | garrison | fmt=4P|contested=uncontested | 518 | +37.591 | 29.65 | +127% | +5.1 | 0.22 | 0.58 | 0.98 |
