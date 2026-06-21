# Clone-Residual Divergence Mine — Findings

_Counterfactual action-diff of top-tier producer-family agents vs bare `producer` and our `v5`, on the IDENTICAL observation each agent saw. Discovery only — gate at n>=100 mirror A/B before shipping._

- Target seats analyzed: **186**; classified **posture-clone** (median send-fraction >= 0.9 AND launches/source <= 1.4): **183**; clone decision-turns: **45541**.

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
| flg | 3 | 1640 | 2P | 148 | 0.50 | 1.00 | 0.04 | — |
| flg | 3 | 1640 | 4P | 499 | 1.00 | 1.00 | 0.07 | ✅ |
| flg | 3 | 1640 | 4P | 499 | 1.00 | 1.00 | 0.05 | ✅ |
| Ender | 4 | 1585 | 4P | 324 | 1.00 | 1.00 | 0.09 | ✅ |
| Ender | 4 | 1585 | 2P | 252 | 1.00 | 1.00 | 0.07 | ✅ |
| Ender | 4 | 1585 | 2P | 131 | 1.00 | 1.00 | 0.13 | ✅ |
| Ender | 4 | 1585 | 4P | 139 | 1.00 | 1.00 | 0.05 | ✅ |
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
| Hober Malloc | 7 | 1543 | 2P | 102 | 1.00 | 1.00 | 0.11 | ✅ |
| Hober Malloc | 7 | 1543 | 2P | 499 | 1.00 | 1.00 | 0.01 | ✅ |
| Hober Malloc | 7 | 1543 | 4P | 286 | 1.00 | 1.00 | 0.07 | ✅ |
| Hober Malloc | 7 | 1543 | 2P | 105 | 1.00 | 1.00 | 0.11 | ✅ |
| Audun Ljone Henriksen | 8 | 1525 | 2P | 174 | 0.96 | 1.00 | 0.10 | ✅ |
| Audun Ljone Henriksen | 8 | 1525 | 2P | 116 | 0.95 | 1.00 | 0.07 | ✅ |
| Vadasz & Ascalon | 9 | 1495 | 2P | 102 | 1.00 | 1.00 | 0.06 | ✅ |
| Vadasz & Ascalon | 9 | 1495 | 2P | 191 | 1.00 | 1.00 | 0.12 | ✅ |
| Yuki Okumura | 10 | 1494 | 2P | 499 | 1.00 | 1.00 | 0.03 | ✅ |
| Yuki Okumura | 10 | 1494 | 2P | 132 | 1.00 | 1.00 | 0.15 | ✅ |
| Yuki Okumura | 10 | 1494 | 2P | 499 | 1.00 | 1.00 | 0.04 | ✅ |
| Yuki Okumura | 10 | 1494 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Yuki Okumura | 10 | 1494 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| typeIIIfairy | 11 | 1492 | 2P | 162 | 1.00 | 1.00 | 0.13 | ✅ |
| typeIIIfairy | 11 | 1492 | 2P | 229 | 1.00 | 1.00 | 0.05 | ✅ |
| typeIIIfairy | 11 | 1492 | 4P | 248 | 1.00 | 1.00 | 0.09 | ✅ |
| typeIIIfairy | 11 | 1492 | 2P | 135 | 1.00 | 1.00 | 0.13 | ✅ |
| moriiiiiiiiim | 12 | 1487 | 2P | 206 | 1.00 | 1.00 | 0.11 | ✅ |
| moriiiiiiiiim | 12 | 1487 | 2P | 152 | 1.00 | 1.00 | 0.08 | ✅ |
| Slawek Biel | 13 | 1487 | 2P | 213 | 1.00 | 1.00 | 0.10 | ✅ |
| Slawek Biel | 13 | 1487 | 2P | 155 | 1.00 | 1.00 | 0.14 | ✅ |
| Slawek Biel | 13 | 1487 | 2P | 499 | 1.00 | 1.00 | 0.09 | ✅ |
| Slawek Biel | 13 | 1487 | 2P | 172 | 1.00 | 1.00 | 0.11 | ✅ |
| Slawek Biel | 13 | 1487 | 2P | 499 | 1.00 | 1.00 | 0.08 | ✅ |
| 213tubo | 14 | 1483 | 2P | 148 | 1.00 | 1.00 | 0.10 | ✅ |
| 213tubo | 14 | 1483 | 2P | 282 | 1.00 | 1.00 | 0.03 | ✅ |
| 213tubo | 14 | 1483 | 2P | 297 | 1.00 | 1.00 | 0.03 | ✅ |
| M & J & M.ver2 | 15 | 1479 | 2P | 434 | 1.00 | 1.00 | 0.04 | ✅ |
| M & J & M.ver2 | 15 | 1479 | 2P | 155 | 1.00 | 1.00 | 0.09 | ✅ |
| M & J & M.ver2 | 15 | 1479 | 2P | 149 | 1.00 | 1.00 | 0.11 | ✅ |
| M & J & M.ver2 | 15 | 1479 | 2P | 127 | 1.00 | 1.00 | 0.06 | ✅ |
| M & J & M.ver2 | 15 | 1479 | 2P | 141 | 1.00 | 1.00 | 0.08 | ✅ |
| M & J & M.ver2 | 15 | 1479 | 2P | 206 | 1.00 | 1.00 | 0.10 | ✅ |
| M & J & M.ver2 | 15 | 1479 | 2P | 229 | 1.00 | 1.00 | 0.11 | ✅ |
| M & J & M.ver2 | 15 | 1479 | 2P | 145 | 1.00 | 1.00 | 0.09 | ✅ |
| M & J & M.ver2 | 15 | 1479 | 2P | 139 | 1.00 | 1.00 | 0.12 | ✅ |
| M & J & M.ver2 | 15 | 1479 | 2P | 499 | 1.00 | 1.00 | 0.12 | ✅ |
| Azat Akhtyamov | 16 | 1476 | 2P | 499 | 1.00 | 1.00 | 0.07 | ✅ |
| Azat Akhtyamov | 16 | 1476 | 4P | 139 | 1.00 | 1.00 | 0.06 | ✅ |
| Azat Akhtyamov | 16 | 1476 | 2P | 118 | 1.00 | 1.00 | 0.12 | ✅ |
| Azat Akhtyamov | 16 | 1476 | 4P | 139 | 1.00 | 1.00 | 0.04 | ✅ |
| Azat Akhtyamov | 16 | 1476 | 2P | 135 | 1.00 | 1.00 | 0.06 | ✅ |
| Azat Akhtyamov | 16 | 1476 | 2P | 172 | 1.00 | 1.00 | 0.01 | ✅ |
| Azat Akhtyamov | 16 | 1476 | 2P | 112 | 1.00 | 1.00 | 0.10 | ✅ |
| Azat Akhtyamov | 16 | 1476 | 2P | 171 | 1.00 | 1.00 | 0.06 | ✅ |
| Azat Akhtyamov | 16 | 1476 | 2P | 120 | 1.00 | 1.00 | 0.05 | ✅ |
| Nebraskinator | 17 | 1464 | 2P | 149 | 1.00 | 1.00 | 0.03 | ✅ |
| Nebraskinator | 17 | 1464 | 4P | 181 | 1.00 | 1.00 | 0.08 | ✅ |
| One Man Wrecking Machine | 18 | 1459 | 4P | 139 | 1.00 | 1.00 | 0.06 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 205 | 1.00 | 1.00 | 0.03 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 200 | 1.00 | 1.00 | 0.08 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 172 | 1.00 | 1.00 | 0.06 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 126 | 1.00 | 1.00 | 0.08 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 150 | 1.00 | 1.00 | 0.11 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 155 | 1.00 | 1.00 | 0.05 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 154 | 1.00 | 1.00 | 0.14 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 107 | 1.00 | 1.00 | 0.17 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 139 | 1.00 | 1.00 | 0.09 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 499 | 1.00 | 1.00 | 0.01 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 142 | 1.00 | 1.00 | 0.12 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 185 | 1.00 | 1.00 | 0.08 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 132 | 1.00 | 1.00 | 0.11 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 134 | 1.00 | 1.00 | 0.17 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 177 | 1.00 | 1.00 | 0.12 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 119 | 1.00 | 1.00 | 0.04 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 152 | 1.00 | 1.00 | 0.06 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 184 | 1.00 | 1.00 | 0.07 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 206 | 1.00 | 1.00 | 0.16 | ✅ |
| Roche Overflow | 19 | 1456 | 2P | 206 | 1.00 | 1.00 | 0.14 | ✅ |
| skalermo | 20 | 1449 | 2P | 86 | 1.00 | 1.00 | 0.04 | ✅ |
| skalermo | 20 | 1449 | 2P | 150 | 1.00 | 1.00 | 0.16 | ✅ |
| skalermo | 20 | 1449 | 4P | 125 | 1.00 | 1.00 | 0.06 | ✅ |
| skalermo | 20 | 1449 | 2P | 145 | 1.00 | 1.00 | 0.12 | ✅ |
| Controlvector | 21 | 1447 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Controlvector | 21 | 1447 | 2P | 134 | 1.00 | 1.00 | 0.03 | ✅ |
| Scool | 22 | 1444 | 2P | 153 | 1.00 | 1.00 | 0.09 | ✅ |
| Scool | 22 | 1444 | 2P | 174 | 1.00 | 1.00 | 0.10 | ✅ |
| Scool | 22 | 1444 | 2P | 166 | 1.00 | 1.00 | 0.09 | ✅ |
| TonyK | 23 | 1440 | 2P | 172 | 1.00 | 1.00 | 0.05 | ✅ |
| TonyK | 23 | 1440 | 2P | 198 | 1.00 | 1.00 | 0.08 | ✅ |
| TonyK | 23 | 1440 | 4P | 175 | 1.00 | 1.00 | 0.18 | ✅ |
| Gregor Lied | 24 | 1437 | 2P | 499 | 1.00 | 1.00 | 0.04 | ✅ |
| Gregor Lied | 24 | 1437 | 2P | 265 | 1.00 | 1.00 | 0.08 | ✅ |
| Gregor Lied | 24 | 1437 | 2P | 230 | 1.00 | 1.00 | 0.10 | ✅ |
| Gregor Lied | 24 | 1437 | 2P | 155 | 1.00 | 1.00 | 0.08 | ✅ |
| Gregor Lied | 24 | 1437 | 2P | 392 | 1.00 | 1.00 | 0.05 | ✅ |
| Gregor Lied | 24 | 1437 | 2P | 301 | 1.00 | 1.00 | 0.07 | ✅ |
| Gregor Lied | 24 | 1437 | 2P | 127 | 1.00 | 1.00 | 0.13 | ✅ |
| Gregor Lied | 24 | 1437 | 2P | 499 | 1.00 | 1.00 | 0.10 | ✅ |
| Gregor Lied | 24 | 1437 | 2P | 499 | 1.00 | 1.00 | 0.05 | ✅ |
| Gregor Lied | 24 | 1437 | 2P | 172 | 1.00 | 1.00 | 0.02 | ✅ |
| Gregor Lied | 24 | 1437 | 2P | 206 | 1.00 | 1.00 | 0.07 | ✅ |
| Gregor Lied | 24 | 1437 | 2P | 204 | 1.00 | 1.00 | 0.04 | ✅ |
| Gregor Lied | 24 | 1437 | 2P | 288 | 1.00 | 1.00 | 0.07 | ✅ |
| Gregor Lied | 24 | 1437 | 2P | 139 | 1.00 | 1.00 | 0.11 | ✅ |
| Piotr Gabrys | 25 | 1436 | 2P | 148 | 1.00 | 1.00 | 0.07 | ✅ |
| Piotr Gabrys | 25 | 1436 | 2P | 434 | 1.00 | 1.00 | 0.06 | ✅ |
| Piotr Gabrys | 25 | 1436 | 2P | 130 | 1.00 | 1.00 | 0.17 | ✅ |
| Piotr Gabrys | 25 | 1436 | 2P | 279 | 1.00 | 1.00 | 0.03 | ✅ |
| Piotr Gabrys | 25 | 1436 | 4P | 126 | 1.00 | 1.00 | 0.05 | ✅ |
| Piotr Gabrys | 25 | 1436 | 2P | 499 | 1.00 | 1.00 | 0.06 | ✅ |
| Piotr Gabrys | 25 | 1436 | 2P | 150 | 1.00 | 1.00 | 0.08 | ✅ |
| Piotr Gabrys | 25 | 1436 | 4P | 125 | 1.00 | 1.00 | 0.02 | ✅ |
| Piotr Gabrys | 25 | 1436 | 2P | 499 | 1.00 | 1.00 | 0.02 | ✅ |
| Piotr Gabrys | 25 | 1436 | 2P | 297 | 1.00 | 1.00 | 0.16 | ✅ |
| Piotr Gabrys | 25 | 1436 | 2P | 184 | 1.00 | 1.00 | 0.17 | ✅ |
| Piotr Gabrys | 25 | 1436 | 2P | 219 | 1.00 | 1.00 | 0.08 | ✅ |
| Boey | 26 | 1436 | 4P | 499 | 1.00 | 1.00 | 0.06 | ✅ |
| Boey | 26 | 1436 | 2P | 184 | 1.00 | 1.06 | 0.07 | ✅ |
| Boey | 26 | 1436 | 4P | 499 | 1.00 | 1.06 | 0.02 | ✅ |
| Boey | 26 | 1436 | 4P | 125 | 1.00 | 1.11 | 0.12 | ✅ |
| bowwowforeach | 27 | 1433 | 2P | 126 | 1.00 | 1.01 | 0.17 | ✅ |
| bowwowforeach | 27 | 1433 | 2P | 120 | 1.00 | 1.01 | 0.11 | ✅ |
| bowwowforeach | 27 | 1433 | 2P | 163 | 1.00 | 1.01 | 0.05 | ✅ |
| bowwowforeach | 27 | 1433 | 2P | 144 | 1.00 | 1.01 | 0.07 | ✅ |
| bowwowforeach | 27 | 1433 | 2P | 99 | 1.00 | 1.00 | 0.13 | ✅ |
| bowwowforeach | 27 | 1433 | 4P | 499 | 1.00 | 1.00 | 0.10 | ✅ |
| bowwowforeach | 27 | 1433 | 4P | 187 | 1.00 | 1.02 | 0.15 | ✅ |
| bowwowforeach | 27 | 1433 | 4P | 499 | 1.00 | 1.02 | 0.07 | ✅ |
| bowwowforeach | 27 | 1433 | 4P | 175 | 1.00 | 1.02 | 0.09 | ✅ |
| bowwowforeach | 27 | 1433 | 4P | 286 | 1.00 | 1.02 | 0.14 | ✅ |
| bowwowforeach | 27 | 1433 | 2P | 252 | 1.00 | 1.01 | 0.11 | ✅ |
| bowwowforeach | 27 | 1433 | 2P | 140 | 1.00 | 1.01 | 0.09 | ✅ |
| bowwowforeach | 27 | 1433 | 2P | 184 | 1.00 | 1.02 | 0.15 | ✅ |
| bowwowforeach | 27 | 1433 | 2P | 105 | 1.00 | 1.01 | 0.15 | ✅ |
| bowwowforeach | 27 | 1433 | 2P | 499 | 1.00 | 1.00 | 0.05 | ✅ |
| bowwowforeach | 27 | 1433 | 4P | 499 | 1.00 | 1.01 | 0.03 | ✅ |
| bowwowforeach | 27 | 1433 | 4P | 125 | 1.00 | 1.01 | 0.02 | ✅ |
| bowwowforeach | 27 | 1433 | 2P | 191 | 1.00 | 1.01 | 0.09 | ✅ |
| bowwowforeach | 27 | 1433 | 2P | 174 | 1.00 | 1.01 | 0.12 | ✅ |
| bowwowforeach | 27 | 1433 | 2P | 206 | 1.00 | 1.02 | 0.05 | ✅ |
| bowwowforeach | 27 | 1433 | 2P | 116 | 1.00 | 1.03 | 0.15 | ✅ |
| bowwowforeach | 27 | 1433 | 2P | 110 | 1.00 | 1.02 | 0.07 | ✅ |
| bowwowforeach | 27 | 1433 | 2P | 152 | 1.00 | 1.04 | 0.14 | ✅ |
| bowwowforeach | 27 | 1433 | 2P | 171 | 1.00 | 1.00 | 0.13 | ✅ |
| Artem | 28 | 1428 | 2P | 230 | 1.00 | 1.00 | 0.17 | ✅ |
| Artem | 28 | 1428 | 2P | 130 | 1.00 | 1.00 | 0.03 | ✅ |
| Artem | 28 | 1428 | 2P | 237 | 1.00 | 1.00 | 0.02 | ✅ |
| Artem | 28 | 1428 | 2P | 117 | 1.00 | 1.00 | 0.13 | ✅ |
| Artem | 28 | 1428 | 2P | 156 | 1.00 | 1.00 | 0.16 | ✅ |
| Artem | 28 | 1428 | 4P | 499 | 1.00 | 1.00 | 0.04 | ✅ |
| Artem | 28 | 1428 | 4P | 142 | 1.00 | 1.00 | 0.01 | ✅ |
| Abhyuday | 29 | 1412 | 2P | 122 | 1.00 | 1.00 | 0.13 | ✅ |
| Abhyuday | 29 | 1412 | 4P | 499 | 1.00 | 1.00 | 0.03 | ✅ |
| Ebi | 30 | 1411 | 2P | 392 | 1.00 | 1.00 | 0.09 | ✅ |
| Ebi | 30 | 1411 | 2P | 169 | 1.00 | 1.00 | 0.14 | ✅ |
| Ebi | 30 | 1411 | 2P | 221 | 1.00 | 1.00 | 0.12 | ✅ |

## Selection divergence (clone vs baseline, same obs)

`pair_prec` = of the clone's launches, fraction producer also makes (low = clone makes launches producer wouldn't = different/extra targets). `pair_rec` = of producer's launches, fraction the clone also makes (low = clone SKIPS launches producer makes). `src_*` = same at source-planet granularity. `tgt_match|src` = on shared source planets, fraction with the SAME target. 1.0 = identical to producer; lower = more divergent.

| vs | metric | state class | n | mean | sd |
|---|---|---|---|---|---|
| prod | pair_prec | all=all | 23830 | 0.173 | 0.326 |
| prod | pair_prec | contested=contested | 3731 | 0.146 | 0.311 |
| prod | pair_prec | contested=uncontested | 20099 | 0.178 | 0.328 |
| prod | pair_prec | fmt=2P | 19107 | 0.183 | 0.332 |
| prod | pair_prec | fmt=2P|contested=uncontested | 19107 | 0.183 | 0.332 |
| prod | pair_prec | fmt=2P|phase=end | 1933 | 0.121 | 0.270 |
| prod | pair_prec | fmt=2P|phase=mid | 12960 | 0.173 | 0.316 |
| prod | pair_prec | fmt=2P|phase=open | 4214 | 0.242 | 0.391 |
| prod | pair_prec | fmt=2P|standing=ahead | 11747 | 0.181 | 0.331 |
| prod | pair_prec | fmt=2P|standing=behind | 7360 | 0.187 | 0.333 |
| prod | pair_prec | fmt=4P | 4723 | 0.131 | 0.296 |
| prod | pair_prec | fmt=4P|contested=contested | 3731 | 0.146 | 0.311 |
| prod | pair_prec | fmt=4P|contested=uncontested | 992 | 0.077 | 0.223 |
| prod | pair_prec | fmt=4P|phase=end | 565 | 0.125 | 0.292 |
| prod | pair_prec | fmt=4P|phase=mid | 3221 | 0.130 | 0.289 |
| prod | pair_prec | fmt=4P|phase=open | 937 | 0.139 | 0.322 |
| prod | pair_prec | fmt=4P|standing=ahead | 1998 | 0.118 | 0.281 |
| prod | pair_prec | fmt=4P|standing=behind | 2725 | 0.141 | 0.306 |
| prod | pair_prec | phase=end | 2498 | 0.122 | 0.275 |
| prod | pair_prec | phase=mid | 16181 | 0.165 | 0.311 |
| prod | pair_prec | phase=open | 5151 | 0.223 | 0.382 |
| prod | pair_prec | standing=ahead | 13745 | 0.172 | 0.325 |
| prod | pair_prec | standing=behind | 10085 | 0.174 | 0.326 |
| prod | pair_rec | all=all | 29709 | 0.113 | 0.256 |
| prod | pair_rec | contested=contested | 4388 | 0.111 | 0.263 |
| prod | pair_rec | contested=uncontested | 25321 | 0.113 | 0.255 |
| prod | pair_rec | fmt=2P | 24033 | 0.116 | 0.257 |
| prod | pair_rec | fmt=2P|contested=uncontested | 24033 | 0.116 | 0.257 |
| prod | pair_rec | fmt=2P|phase=end | 3311 | 0.041 | 0.146 |
| prod | pair_rec | fmt=2P|phase=mid | 16543 | 0.107 | 0.237 |
| prod | pair_rec | fmt=2P|phase=open | 4179 | 0.208 | 0.358 |
| prod | pair_rec | fmt=2P|standing=ahead | 15007 | 0.110 | 0.252 |
| prod | pair_rec | fmt=2P|standing=behind | 9026 | 0.124 | 0.265 |
| prod | pair_rec | fmt=4P | 5676 | 0.100 | 0.251 |
| prod | pair_rec | fmt=4P|contested=contested | 4388 | 0.111 | 0.263 |
| prod | pair_rec | fmt=4P|contested=uncontested | 1288 | 0.064 | 0.200 |
| prod | pair_rec | fmt=4P|phase=end | 902 | 0.053 | 0.173 |
| prod | pair_rec | fmt=4P|phase=mid | 3831 | 0.105 | 0.251 |
| prod | pair_rec | fmt=4P|phase=open | 943 | 0.128 | 0.302 |
| prod | pair_rec | fmt=4P|standing=ahead | 2378 | 0.097 | 0.244 |
| prod | pair_rec | fmt=4P|standing=behind | 3298 | 0.103 | 0.256 |
| prod | pair_rec | phase=end | 4213 | 0.043 | 0.152 |
| prod | pair_rec | phase=mid | 20374 | 0.107 | 0.240 |
| prod | pair_rec | phase=open | 5122 | 0.194 | 0.350 |
| prod | pair_rec | standing=ahead | 17385 | 0.109 | 0.251 |
| prod | pair_rec | standing=behind | 12324 | 0.119 | 0.263 |
| prod | src_prec | all=all | 23830 | 0.393 | 0.426 |
| prod | src_prec | contested=contested | 3731 | 0.329 | 0.414 |
| prod | src_prec | contested=uncontested | 20099 | 0.405 | 0.427 |
| prod | src_prec | fmt=2P | 19107 | 0.417 | 0.429 |
| prod | src_prec | fmt=2P|contested=uncontested | 19107 | 0.417 | 0.429 |
| prod | src_prec | fmt=2P|phase=end | 1933 | 0.412 | 0.441 |
| prod | src_prec | fmt=2P|phase=mid | 12960 | 0.419 | 0.421 |
| prod | src_prec | fmt=2P|phase=open | 4214 | 0.414 | 0.447 |
| prod | src_prec | fmt=2P|standing=ahead | 11747 | 0.432 | 0.435 |
| prod | src_prec | fmt=2P|standing=behind | 7360 | 0.394 | 0.419 |
| prod | src_prec | fmt=4P | 4723 | 0.297 | 0.401 |
| prod | src_prec | fmt=4P|contested=contested | 3731 | 0.329 | 0.414 |
| prod | src_prec | fmt=4P|contested=uncontested | 992 | 0.174 | 0.317 |
| prod | src_prec | fmt=4P|phase=end | 565 | 0.220 | 0.368 |
| prod | src_prec | fmt=4P|phase=mid | 3221 | 0.304 | 0.395 |
| prod | src_prec | fmt=4P|phase=open | 937 | 0.318 | 0.434 |
| prod | src_prec | fmt=4P|standing=ahead | 1998 | 0.265 | 0.381 |
| prod | src_prec | fmt=4P|standing=behind | 2725 | 0.319 | 0.413 |
| prod | src_prec | phase=end | 2498 | 0.368 | 0.433 |
| prod | src_prec | phase=mid | 16181 | 0.396 | 0.418 |
| prod | src_prec | phase=open | 5151 | 0.397 | 0.446 |
| prod | src_prec | standing=ahead | 13745 | 0.408 | 0.431 |
| prod | src_prec | standing=behind | 10085 | 0.374 | 0.419 |
| prod | src_rec | all=all | 29709 | 0.248 | 0.348 |
| prod | src_rec | contested=contested | 4388 | 0.255 | 0.370 |
| prod | src_rec | contested=uncontested | 25321 | 0.247 | 0.344 |
| prod | src_rec | fmt=2P | 24033 | 0.252 | 0.346 |
| prod | src_rec | fmt=2P|contested=uncontested | 24033 | 0.252 | 0.346 |
| prod | src_rec | fmt=2P|phase=end | 3311 | 0.121 | 0.238 |
| prod | src_rec | fmt=2P|phase=mid | 16543 | 0.255 | 0.340 |
| prod | src_rec | fmt=2P|phase=open | 4179 | 0.346 | 0.404 |
| prod | src_rec | fmt=2P|standing=ahead | 15007 | 0.245 | 0.339 |
| prod | src_rec | fmt=2P|standing=behind | 9026 | 0.264 | 0.356 |
| prod | src_rec | fmt=4P | 5676 | 0.231 | 0.358 |
| prod | src_rec | fmt=4P|contested=contested | 4388 | 0.255 | 0.370 |
| prod | src_rec | fmt=4P|contested=uncontested | 1288 | 0.147 | 0.295 |
| prod | src_rec | fmt=4P|phase=end | 902 | 0.096 | 0.230 |
| prod | src_rec | fmt=4P|phase=mid | 3831 | 0.248 | 0.362 |
| prod | src_rec | fmt=4P|phase=open | 943 | 0.290 | 0.407 |
| prod | src_rec | fmt=4P|standing=ahead | 2378 | 0.226 | 0.353 |
| prod | src_rec | fmt=4P|standing=behind | 3298 | 0.234 | 0.361 |
| prod | src_rec | phase=end | 4213 | 0.116 | 0.236 |
| prod | src_rec | phase=mid | 20374 | 0.254 | 0.344 |
| prod | src_rec | phase=open | 5122 | 0.336 | 0.405 |
| prod | src_rec | standing=ahead | 17385 | 0.243 | 0.341 |
| prod | src_rec | standing=behind | 12324 | 0.256 | 0.358 |
| prod | tgt_match|src | all=all | 12973 | 0.432 | 0.459 |
| prod | tgt_match|src | contested=contested | 1698 | 0.434 | 0.471 |
| prod | tgt_match|src | contested=uncontested | 11275 | 0.431 | 0.457 |
| prod | tgt_match|src | fmt=2P | 10951 | 0.432 | 0.457 |
| prod | tgt_match|src | fmt=2P|contested=uncontested | 10951 | 0.432 | 0.457 |
| prod | tgt_match|src | fmt=2P|phase=end | 1032 | 0.315 | 0.421 |
| prod | tgt_match|src | fmt=2P|phase=mid | 7782 | 0.410 | 0.449 |
| prod | tgt_match|src | fmt=2P|phase=open | 2137 | 0.567 | 0.471 |
| prod | tgt_match|src | fmt=2P|standing=ahead | 6957 | 0.412 | 0.451 |
| prod | tgt_match|src | fmt=2P|standing=behind | 3994 | 0.466 | 0.464 |
| prod | tgt_match|src | fmt=4P | 2022 | 0.430 | 0.470 |
| prod | tgt_match|src | fmt=4P|contested=contested | 1698 | 0.434 | 0.471 |
| prod | tgt_match|src | fmt=4P|contested=uncontested | 324 | 0.409 | 0.460 |
| prod | tgt_match|src | fmt=4P|phase=end | 173 | 0.549 | 0.480 |
| prod | tgt_match|src | fmt=4P|phase=mid | 1486 | 0.417 | 0.463 |
| prod | tgt_match|src | fmt=4P|phase=open | 363 | 0.428 | 0.483 |
| prod | tgt_match|src | fmt=4P|standing=ahead | 851 | 0.412 | 0.460 |
| prod | tgt_match|src | fmt=4P|standing=behind | 1171 | 0.444 | 0.476 |
| prod | tgt_match|src | phase=end | 1205 | 0.348 | 0.437 |
| prod | tgt_match|src | phase=mid | 9268 | 0.411 | 0.451 |
| prod | tgt_match|src | phase=open | 2500 | 0.547 | 0.475 |
| prod | tgt_match|src | standing=ahead | 7808 | 0.412 | 0.452 |
| prod | tgt_match|src | standing=behind | 5165 | 0.461 | 0.467 |
| v5 | pair_prec | all=all | 23830 | 0.168 | 0.323 |
| v5 | pair_prec | contested=contested | 3731 | 0.144 | 0.309 |
| v5 | pair_prec | contested=uncontested | 20099 | 0.173 | 0.325 |
| v5 | pair_prec | fmt=2P | 19107 | 0.178 | 0.329 |
| v5 | pair_prec | fmt=2P|contested=uncontested | 19107 | 0.178 | 0.329 |
| v5 | pair_prec | fmt=2P|phase=end | 1933 | 0.118 | 0.274 |
| v5 | pair_prec | fmt=2P|phase=mid | 12960 | 0.169 | 0.313 |
| v5 | pair_prec | fmt=2P|phase=open | 4214 | 0.235 | 0.388 |
| v5 | pair_prec | fmt=2P|standing=ahead | 11747 | 0.179 | 0.331 |
| v5 | pair_prec | fmt=2P|standing=behind | 7360 | 0.177 | 0.327 |
| v5 | pair_prec | fmt=4P | 4723 | 0.129 | 0.294 |
| v5 | pair_prec | fmt=4P|contested=contested | 3731 | 0.144 | 0.309 |
| v5 | pair_prec | fmt=4P|contested=uncontested | 992 | 0.073 | 0.218 |
| v5 | pair_prec | fmt=4P|phase=end | 565 | 0.115 | 0.281 |
| v5 | pair_prec | fmt=4P|phase=mid | 3221 | 0.129 | 0.288 |
| v5 | pair_prec | fmt=4P|phase=open | 937 | 0.138 | 0.321 |
| v5 | pair_prec | fmt=4P|standing=ahead | 1998 | 0.115 | 0.278 |
| v5 | pair_prec | fmt=4P|standing=behind | 2725 | 0.139 | 0.305 |
| v5 | pair_prec | phase=end | 2498 | 0.118 | 0.276 |
| v5 | pair_prec | phase=mid | 16181 | 0.161 | 0.309 |
| v5 | pair_prec | phase=open | 5151 | 0.217 | 0.379 |
| v5 | pair_prec | standing=ahead | 13745 | 0.170 | 0.324 |
| v5 | pair_prec | standing=behind | 10085 | 0.166 | 0.322 |
| v5 | pair_rec | all=all | 28086 | 0.125 | 0.273 |
| v5 | pair_rec | contested=contested | 4140 | 0.123 | 0.280 |
| v5 | pair_rec | contested=uncontested | 23946 | 0.125 | 0.271 |
| v5 | pair_rec | fmt=2P | 22911 | 0.128 | 0.273 |
| v5 | pair_rec | fmt=2P|contested=uncontested | 22911 | 0.128 | 0.273 |
| v5 | pair_rec | fmt=2P|phase=end | 3109 | 0.048 | 0.161 |
| v5 | pair_rec | fmt=2P|phase=mid | 15801 | 0.120 | 0.256 |
| v5 | pair_rec | fmt=2P|phase=open | 4001 | 0.220 | 0.368 |
| v5 | pair_rec | fmt=2P|standing=ahead | 14626 | 0.120 | 0.265 |
| v5 | pair_rec | fmt=2P|standing=behind | 8285 | 0.140 | 0.287 |
| v5 | pair_rec | fmt=4P | 5175 | 0.113 | 0.269 |
| v5 | pair_rec | fmt=4P|contested=contested | 4140 | 0.123 | 0.280 |
| v5 | pair_rec | fmt=4P|contested=uncontested | 1035 | 0.076 | 0.216 |
| v5 | pair_rec | fmt=4P|phase=end | 736 | 0.060 | 0.180 |
| v5 | pair_rec | fmt=4P|phase=mid | 3573 | 0.118 | 0.270 |
| v5 | pair_rec | fmt=4P|phase=open | 866 | 0.141 | 0.316 |
| v5 | pair_rec | fmt=4P|standing=ahead | 2314 | 0.103 | 0.254 |
| v5 | pair_rec | fmt=4P|standing=behind | 2861 | 0.122 | 0.280 |
| v5 | pair_rec | phase=end | 3845 | 0.050 | 0.165 |
| v5 | pair_rec | phase=mid | 19374 | 0.119 | 0.259 |
| v5 | pair_rec | phase=open | 4867 | 0.206 | 0.360 |
| v5 | pair_rec | standing=ahead | 16940 | 0.118 | 0.264 |
| v5 | pair_rec | standing=behind | 11146 | 0.136 | 0.286 |
| v5 | src_prec | all=all | 23830 | 0.350 | 0.416 |
| v5 | src_prec | contested=contested | 3731 | 0.298 | 0.402 |
| v5 | src_prec | contested=uncontested | 20099 | 0.360 | 0.418 |
| v5 | src_prec | fmt=2P | 19107 | 0.370 | 0.420 |
| v5 | src_prec | fmt=2P|contested=uncontested | 19107 | 0.370 | 0.420 |
| v5 | src_prec | fmt=2P|phase=end | 1933 | 0.345 | 0.433 |
| v5 | src_prec | fmt=2P|phase=mid | 12960 | 0.370 | 0.411 |
| v5 | src_prec | fmt=2P|phase=open | 4214 | 0.383 | 0.441 |
| v5 | src_prec | fmt=2P|standing=ahead | 11747 | 0.398 | 0.429 |
| v5 | src_prec | fmt=2P|standing=behind | 7360 | 0.325 | 0.402 |
| v5 | src_prec | fmt=4P | 4723 | 0.271 | 0.389 |
| v5 | src_prec | fmt=4P|contested=contested | 3731 | 0.298 | 0.402 |
| v5 | src_prec | fmt=4P|contested=uncontested | 992 | 0.168 | 0.315 |
| v5 | src_prec | fmt=4P|phase=end | 565 | 0.201 | 0.356 |
| v5 | src_prec | fmt=4P|phase=mid | 3221 | 0.280 | 0.383 |
| v5 | src_prec | fmt=4P|phase=open | 937 | 0.283 | 0.420 |
| v5 | src_prec | fmt=4P|standing=ahead | 1998 | 0.247 | 0.371 |
| v5 | src_prec | fmt=4P|standing=behind | 2725 | 0.288 | 0.400 |
| v5 | src_prec | phase=end | 2498 | 0.313 | 0.421 |
| v5 | src_prec | phase=mid | 16181 | 0.352 | 0.407 |
| v5 | src_prec | phase=open | 5151 | 0.364 | 0.439 |
| v5 | src_prec | standing=ahead | 13745 | 0.376 | 0.424 |
| v5 | src_prec | standing=behind | 10085 | 0.315 | 0.402 |
| v5 | src_rec | all=all | 28086 | 0.255 | 0.358 |
| v5 | src_rec | contested=contested | 4140 | 0.262 | 0.380 |
| v5 | src_rec | contested=uncontested | 23946 | 0.254 | 0.354 |
| v5 | src_rec | fmt=2P | 22911 | 0.257 | 0.356 |
| v5 | src_rec | fmt=2P|contested=uncontested | 22911 | 0.257 | 0.356 |
| v5 | src_rec | fmt=2P|phase=end | 3109 | 0.122 | 0.246 |
| v5 | src_rec | fmt=2P|phase=mid | 15801 | 0.259 | 0.350 |
| v5 | src_rec | fmt=2P|phase=open | 4001 | 0.353 | 0.412 |
| v5 | src_rec | fmt=2P|standing=ahead | 14626 | 0.255 | 0.351 |
| v5 | src_rec | fmt=2P|standing=behind | 8285 | 0.261 | 0.364 |
| v5 | src_rec | fmt=4P | 5175 | 0.246 | 0.370 |
| v5 | src_rec | fmt=4P|contested=contested | 4140 | 0.262 | 0.380 |
| v5 | src_rec | fmt=4P|contested=uncontested | 1035 | 0.180 | 0.318 |
| v5 | src_rec | fmt=4P|phase=end | 736 | 0.109 | 0.240 |
| v5 | src_rec | fmt=4P|phase=mid | 3573 | 0.261 | 0.374 |
| v5 | src_rec | fmt=4P|phase=open | 866 | 0.297 | 0.416 |
| v5 | src_rec | fmt=4P|standing=ahead | 2314 | 0.230 | 0.359 |
| v5 | src_rec | fmt=4P|standing=behind | 2861 | 0.258 | 0.378 |
| v5 | src_rec | phase=end | 3845 | 0.120 | 0.245 |
| v5 | src_rec | phase=mid | 19374 | 0.260 | 0.355 |
| v5 | src_rec | phase=open | 4867 | 0.343 | 0.413 |
| v5 | src_rec | standing=ahead | 16940 | 0.251 | 0.352 |
| v5 | src_rec | standing=behind | 11146 | 0.261 | 0.368 |
| v5 | tgt_match|src | all=all | 11859 | 0.471 | 0.467 |
| v5 | tgt_match|src | contested=contested | 1578 | 0.468 | 0.477 |
| v5 | tgt_match|src | contested=uncontested | 10281 | 0.472 | 0.465 |
| v5 | tgt_match|src | fmt=2P | 9966 | 0.474 | 0.465 |
| v5 | tgt_match|src | fmt=2P|contested=uncontested | 9966 | 0.474 | 0.465 |
| v5 | tgt_match|src | fmt=2P|phase=end | 865 | 0.360 | 0.440 |
| v5 | tgt_match|src | fmt=2P|phase=mid | 7095 | 0.453 | 0.460 |
| v5 | tgt_match|src | fmt=2P|phase=open | 2006 | 0.598 | 0.470 |
| v5 | tgt_match|src | fmt=2P|standing=ahead | 6545 | 0.444 | 0.460 |
| v5 | tgt_match|src | fmt=2P|standing=behind | 3421 | 0.531 | 0.470 |
| v5 | tgt_match|src | fmt=4P | 1893 | 0.456 | 0.475 |
| v5 | tgt_match|src | fmt=4P|contested=contested | 1578 | 0.468 | 0.477 |
| v5 | tgt_match|src | fmt=4P|contested=uncontested | 315 | 0.397 | 0.459 |
| v5 | tgt_match|src | fmt=4P|phase=end | 160 | 0.552 | 0.483 |
| v5 | tgt_match|src | fmt=4P|phase=mid | 1406 | 0.442 | 0.469 |
| v5 | tgt_match|src | fmt=4P|phase=open | 327 | 0.473 | 0.489 |
| v5 | tgt_match|src | fmt=4P|standing=ahead | 812 | 0.428 | 0.464 |
| v5 | tgt_match|src | fmt=4P|standing=behind | 1081 | 0.478 | 0.482 |
| v5 | tgt_match|src | phase=end | 1025 | 0.390 | 0.452 |
| v5 | tgt_match|src | phase=mid | 8501 | 0.451 | 0.461 |
| v5 | tgt_match|src | phase=open | 2333 | 0.580 | 0.475 |
| v5 | tgt_match|src | standing=ahead | 7357 | 0.442 | 0.460 |
| v5 | tgt_match|src | standing=behind | 4502 | 0.518 | 0.473 |

## Ranked behavioural divergences (clone − baseline)

Positive `Δ` = clone does MORE of the axis than the baseline on the SAME obs. `base` = baseline's own mean (for scale); `pct` = Δ as % of base; `t` = t-stat (|t|>=3 kept); `eff` = |Δ|/sd; `cons` = sign-agreement; `nz` = fraction of turns exercising the axis. Ranked by |t|·eff.

| # | vs | axis | state class | n | Δ | base | pct | t | eff | cons | nz |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | prod | active | phase=end | 6063 | -0.283 | 0.69 | -41% | -37.3 | 0.48 | 0.36 | 0.43 |
| 2 | prod | attack_rate | fmt=2P|phase=end | 1559 | -0.297 | 0.54 | -55% | -27.9 | 0.71 | 0.73 | 0.91 |
| 3 | prod | active | fmt=2P|phase=end | 4126 | -0.334 | 0.80 | -42% | -33.8 | 0.53 | 0.42 | 0.52 |
| 4 | prod | attack_rate | phase=end | 2057 | -0.254 | 0.48 | -52% | -27.1 | 0.60 | 0.69 | 0.87 |
| 5 | prod | waves | fmt=2P|standing=behind | 13418 | -0.678 | 1.73 | -39% | -39.5 | 0.34 | 0.44 | 0.67 |
| 6 | prod | sources | fmt=2P|standing=behind | 13418 | -0.676 | 1.73 | -39% | -39.5 | 0.34 | 0.44 | 0.67 |
| 7 | prod | active | contested=uncontested | 35909 | -0.145 | 0.71 | -21% | -46.7 | 0.25 | 0.26 | 0.37 |
| 8 | prod | active | fmt=2P|phase=mid | 21190 | -0.169 | 0.78 | -22% | -42.1 | 0.29 | 0.27 | 0.37 |
| 9 | prod | waves | fmt=2P|phase=end | 4126 | -2.640 | 4.23 | -62% | -30.4 | 0.47 | 0.69 | 0.84 |
| 10 | prod | sources | fmt=2P|phase=end | 4126 | -2.627 | 4.21 | -62% | -30.3 | 0.47 | 0.69 | 0.84 |
| 11 | prod | active | all=all | 45541 | -0.129 | 0.65 | -20% | -47.9 | 0.22 | 0.24 | 0.35 |
| 12 | prod | active | fmt=2P | 34016 | -0.145 | 0.71 | -20% | -45.3 | 0.25 | 0.26 | 0.37 |
| 13 | prod | active | fmt=2P|contested=uncontested | 34016 | -0.145 | 0.71 | -20% | -45.3 | 0.25 | 0.26 | 0.37 |
| 14 | prod | active | phase=mid | 28498 | -0.147 | 0.71 | -21% | -43.6 | 0.26 | 0.25 | 0.35 |
| 15 | prod | tgt_dist | fmt=2P|phase=end | 1498 | -9.579 | 28.58 | -34% | -24.2 | 0.62 | 0.81 | 0.99 |
| 16 | prod | ships | contested=uncontested | 35909 | -462.063 | 661.56 | -70% | -45.2 | 0.24 | 0.54 | 0.78 |
| 17 | prod | sources | standing=behind | 21420 | -0.485 | 1.36 | -36% | -40.9 | 0.28 | 0.37 | 0.57 |
| 18 | prod | waves | standing=behind | 21420 | -0.486 | 1.36 | -36% | -40.9 | 0.28 | 0.37 | 0.57 |
| 19 | prod | ships | phase=end | 6063 | -1557.375 | 2028.83 | -77% | -31.9 | 0.41 | 0.62 | 0.76 |
| 20 | prod | ships | all=all | 45541 | -377.777 | 541.01 | -70% | -46.5 | 0.22 | 0.50 | 0.73 |
| 21 | prod | ships | fmt=2P|phase=end | 4126 | -2013.879 | 2668.46 | -75% | -29.4 | 0.46 | 0.72 | 0.89 |
| 22 | prod | ships | standing=behind | 21420 | -138.110 | 171.23 | -81% | -40.2 | 0.27 | 0.44 | 0.65 |
| 23 | prod | attack_rate | fmt=2P|standing=ahead | 9527 | -0.147 | 0.36 | -41% | -34.2 | 0.35 | 0.49 | 0.69 |
| 24 | prod | ships | standing=ahead | 24121 | -590.606 | 869.38 | -68% | -39.7 | 0.26 | 0.55 | 0.80 |
| 25 | prod | active | fmt=2P|standing=ahead | 20598 | -0.158 | 0.73 | -22% | -38.5 | 0.27 | 0.27 | 0.37 |
| 26 | prod | active | standing=ahead | 24121 | -0.151 | 0.72 | -21% | -39.5 | 0.25 | 0.26 | 0.37 |
| 27 | prod | ships | fmt=2P | 34016 | -443.063 | 643.47 | -69% | -41.9 | 0.23 | 0.54 | 0.78 |
| 28 | prod | ships | fmt=2P|contested=uncontested | 34016 | -443.063 | 643.47 | -69% | -41.9 | 0.23 | 0.54 | 0.78 |
| 29 | prod | waves | phase=end | 6063 | -1.873 | 3.33 | -56% | -29.9 | 0.38 | 0.56 | 0.71 |
| 30 | prod | sources | phase=end | 6063 | -1.863 | 3.32 | -56% | -29.8 | 0.38 | 0.56 | 0.71 |
| 31 | v5 | active | fmt=2P|phase=end | 4126 | -0.285 | 0.75 | -38% | -27.5 | 0.43 | 0.40 | 0.52 |
| 32 | prod | ships | fmt=2P|standing=behind | 13418 | -182.395 | 222.80 | -82% | -34.6 | 0.30 | 0.51 | 0.75 |
| 33 | v5 | ships | contested=uncontested | 35909 | -360.632 | 560.13 | -64% | -40.4 | 0.21 | 0.48 | 0.77 |
| 34 | prod | ships | phase=mid | 28498 | -267.854 | 419.71 | -64% | -38.6 | 0.23 | 0.54 | 0.79 |
| 35 | v5 | ships | all=all | 45541 | -296.369 | 459.60 | -64% | -41.8 | 0.20 | 0.45 | 0.71 |
| 36 | v5 | active | phase=end | 6063 | -0.222 | 0.63 | -35% | -28.7 | 0.37 | 0.32 | 0.41 |
| 37 | prod | ships | fmt=2P|standing=ahead | 20598 | -612.868 | 917.51 | -67% | -36.0 | 0.25 | 0.56 | 0.80 |
| 38 | prod | attack_rate | standing=ahead | 11055 | -0.129 | 0.34 | -38% | -32.1 | 0.31 | 0.47 | 0.68 |
| 39 | v5 | ships | standing=ahead | 24121 | -482.668 | 761.45 | -63% | -36.9 | 0.24 | 0.51 | 0.79 |
| 40 | v5 | ships | fmt=2P | 34016 | -351.140 | 551.55 | -64% | -37.9 | 0.21 | 0.49 | 0.77 |
| 41 | v5 | ships | fmt=2P|contested=uncontested | 34016 | -351.140 | 551.55 | -64% | -37.9 | 0.21 | 0.49 | 0.77 |
| 42 | prod | tgt_dist | phase=end | 1982 | -7.657 | 27.11 | -28% | -21.9 | 0.49 | 0.76 | 0.99 |
| 43 | prod | ships | fmt=2P|phase=mid | 21190 | -314.660 | 499.29 | -63% | -34.6 | 0.24 | 0.59 | 0.86 |
| 44 | v5 | ships | standing=behind | 21420 | -86.579 | 119.70 | -72% | -34.6 | 0.24 | 0.37 | 0.62 |
| 45 | prod | attack_rate | fmt=2P | 15304 | -0.114 | 0.35 | -33% | -32.5 | 0.26 | 0.44 | 0.66 |
| 46 | prod | attack_rate | fmt=2P|contested=uncontested | 15304 | -0.114 | 0.35 | -33% | -32.5 | 0.26 | 0.44 | 0.66 |
| 47 | prod | attack_rate | contested=uncontested | 16078 | -0.111 | 0.34 | -33% | -32.6 | 0.26 | 0.44 | 0.66 |
| 48 | v5 | ships | fmt=2P|phase=end | 4126 | -1470.096 | 2124.68 | -69% | -25.1 | 0.39 | 0.65 | 0.87 |
| 49 | v5 | ships | phase=end | 6063 | -1115.234 | 1586.69 | -70% | -27.0 | 0.35 | 0.55 | 0.72 |
| 50 | v5 | ships | fmt=2P|standing=ahead | 20598 | -502.324 | 806.97 | -62% | -33.5 | 0.23 | 0.52 | 0.79 |
| 51 | v5 | active | fmt=2P|standing=ahead | 20598 | -0.140 | 0.71 | -20% | -33.4 | 0.23 | 0.26 | 0.38 |
| 52 | prod | ships | fmt=4P|contested=uncontested | 1893 | -803.474 | 986.54 | -81% | -21.1 | 0.48 | 0.57 | 0.79 |
| 53 | v5 | ships | fmt=2P|standing=behind | 13418 | -119.058 | 159.46 | -75% | -30.9 | 0.27 | 0.43 | 0.74 |
| 54 | prod | ships | fmt=4P|standing=behind | 8002 | -63.852 | 84.77 | -75% | -28.0 | 0.31 | 0.32 | 0.48 |
| 55 | v5 | active | standing=ahead | 24121 | -0.132 | 0.70 | -19% | -34.2 | 0.22 | 0.26 | 0.38 |
| 56 | prod | tgt_prod | all=all | 18116 | +0.320 | 2.82 | +11% | +32.4 | 0.24 | 0.50 | 0.81 |
| 57 | prod | attack_rate | all=all | 18844 | -0.105 | 0.34 | -30% | -32.4 | 0.24 | 0.43 | 0.65 |
| 58 | v5 | ships | phase=mid | 28498 | -233.750 | 385.60 | -61% | -34.8 | 0.21 | 0.49 | 0.78 |
| 59 | v5 | active | fmt=2P|phase=mid | 21190 | -0.134 | 0.75 | -18% | -32.4 | 0.22 | 0.26 | 0.38 |
| 60 | prod | full_rate | phase=end | 2057 | -0.137 | 0.97 | -14% | -20.4 | 0.45 | 0.28 | 0.36 |

## Target-disagreement profile (clone target − producer target)

_Plan 1, the confound-free residual: on the **shared source planets** where the clone and bare `producer` BOTH launch but resolve to **different targets**, what is systematically different about the target the clone picks vs the one producer's flow-diff argmax picks?_

- Shared-source decisions: **17798**; of which the clone aimed elsewhere than producer: **9508** (**53%** disagreement rate — matches `tgt_match|src`).

Each row = a target-feature axis conditioned on a state class. `Δ` = mean(clone_target_feature − producer_target_feature) over disagreements (positive = clone's chosen target scores HIGHER on the axis). `base` = producer-target mean (scale); `pct` = Δ as % of base; `t` = t-stat (|t|>=3 kept); `eff` = |Δ|/sd; `cons` = sign-agreement; `nz` = fraction of disagreements where the axis differs. Axes: `dist` (source→target euclidean), `prod`, `garrison`, `contested` (>=2 owners inbound), `orbiting`, and owner-class indicators `is_enemy`/`is_neutral`/`is_own`. Ranked by |t|·eff.

| # | axis | state class | n | Δ | base | pct | t | eff | cons | nz |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | dist | fmt=2P|phase=end | 1008 | -20.064 | 38.54 | -52% | -26.9 | 0.85 | 0.83 | 1.00 |
| 2 | dist | phase=end | 1083 | -18.847 | 37.69 | -50% | -25.9 | 0.79 | 0.81 | 1.00 |
| 3 | dist | fmt=2P | 8238 | -11.319 | 33.91 | -33% | -39.0 | 0.43 | 0.71 | 1.00 |
| 4 | dist | fmt=2P|contested=uncontested | 8238 | -11.319 | 33.91 | -33% | -39.0 | 0.43 | 0.71 | 1.00 |
| 5 | dist | contested=uncontested | 8477 | -11.258 | 33.80 | -33% | -39.2 | 0.43 | 0.71 | 1.00 |
| 6 | is_own | fmt=2P|standing=ahead | 5648 | +0.312 | 0.38 | +82% | +35.3 | 0.47 | 0.43 | 0.54 |
| 7 | dist | fmt=2P|standing=ahead | 5648 | -12.253 | 34.60 | -35% | -35.0 | 0.47 | 0.72 | 1.00 |
| 8 | is_enemy | fmt=2P|standing=ahead | 5648 | -0.289 | 0.53 | -54% | -33.8 | 0.45 | 0.39 | 0.50 |
| 9 | dist | all=all | 9508 | -10.223 | 33.19 | -31% | -37.1 | 0.38 | 0.69 | 1.00 |
| 10 | is_own | fmt=2P | 8238 | +0.261 | 0.40 | +64% | +35.7 | 0.39 | 0.38 | 0.51 |
| 11 | is_own | fmt=2P|contested=uncontested | 8238 | +0.261 | 0.40 | +64% | +35.7 | 0.39 | 0.38 | 0.51 |
| 12 | dist | standing=ahead | 6236 | -11.329 | 33.98 | -33% | -33.5 | 0.42 | 0.71 | 1.00 |
| 13 | is_own | contested=uncontested | 8477 | +0.253 | 0.41 | +61% | +35.2 | 0.38 | 0.38 | 0.50 |
| 14 | is_own | standing=ahead | 6236 | +0.280 | 0.41 | +69% | +33.1 | 0.42 | 0.40 | 0.52 |
| 15 | is_enemy | fmt=2P | 8238 | -0.240 | 0.51 | -47% | -33.9 | 0.37 | 0.35 | 0.47 |
| 16 | is_enemy | fmt=2P|contested=uncontested | 8238 | -0.240 | 0.51 | -47% | -33.9 | 0.37 | 0.35 | 0.47 |
| 17 | is_enemy | standing=ahead | 6236 | -0.259 | 0.51 | -51% | -31.9 | 0.40 | 0.37 | 0.48 |
| 18 | is_enemy | contested=uncontested | 8477 | -0.233 | 0.50 | -46% | -33.5 | 0.36 | 0.35 | 0.46 |
| 19 | is_own | all=all | 9508 | +0.230 | 0.43 | +54% | +33.8 | 0.35 | 0.36 | 0.50 |
| 20 | dist | fmt=2P|phase=mid | 6135 | -10.428 | 33.64 | -31% | -30.2 | 0.39 | 0.70 | 1.00 |
| 21 | is_enemy | all=all | 9508 | -0.214 | 0.50 | -43% | -32.5 | 0.33 | 0.34 | 0.46 |
| 22 | is_own | fmt=2P|phase=mid | 6135 | +0.250 | 0.42 | +59% | +29.2 | 0.37 | 0.38 | 0.51 |
| 23 | is_enemy | fmt=2P|phase=end | 1008 | -0.395 | 0.67 | -59% | -19.6 | 0.62 | 0.48 | 0.56 |
| 24 | is_enemy | fmt=2P|phase=mid | 6135 | -0.237 | 0.52 | -45% | -28.5 | 0.36 | 0.36 | 0.48 |
| 25 | dist | phase=mid | 7128 | -9.334 | 32.92 | -28% | -28.5 | 0.34 | 0.68 | 1.00 |
| 26 | is_own | fmt=2P|phase=end | 1008 | +0.387 | 0.32 | +120% | +19.1 | 0.60 | 0.48 | 0.57 |
| 27 | is_own | phase=mid | 7128 | +0.217 | 0.45 | +48% | +27.3 | 0.32 | 0.36 | 0.50 |
| 28 | is_enemy | phase=end | 1083 | -0.365 | 0.65 | -56% | -18.5 | 0.56 | 0.46 | 0.55 |
| 29 | is_enemy | phase=mid | 7128 | -0.207 | 0.50 | -41% | -26.9 | 0.32 | 0.34 | 0.46 |
| 30 | is_own | phase=end | 1083 | +0.355 | 0.35 | +102% | +17.8 | 0.54 | 0.46 | 0.56 |
| 31 | dist | fmt=2P|standing=behind | 2590 | -9.282 | 32.41 | -29% | -17.9 | 0.35 | 0.69 | 1.00 |
| 32 | dist | standing=behind | 3272 | -8.115 | 31.68 | -26% | -17.2 | 0.30 | 0.67 | 1.00 |
| 33 | dist | fmt=2P|phase=open | 1095 | -8.260 | 31.16 | -27% | -11.9 | 0.36 | 0.69 | 1.00 |
| 34 | dist | phase=open | 1297 | -7.906 | 30.90 | -26% | -12.2 | 0.34 | 0.68 | 1.00 |
| 35 | is_own | phase=open | 1297 | +0.201 | 0.38 | +53% | +11.5 | 0.32 | 0.32 | 0.44 |
| 36 | is_own | fmt=2P|phase=open | 1095 | +0.202 | 0.38 | +53% | +10.8 | 0.33 | 0.31 | 0.42 |
| 37 | is_own | fmt=2P|standing=behind | 2590 | +0.147 | 0.46 | +32% | +11.6 | 0.23 | 0.29 | 0.44 |
| 38 | is_own | standing=behind | 3272 | +0.137 | 0.46 | +29% | +12.0 | 0.21 | 0.29 | 0.44 |
| 39 | prod | phase=mid | 7128 | +0.317 | 2.78 | +11% | +13.4 | 0.16 | 0.45 | 0.75 |
| 40 | prod | all=all | 9508 | +0.288 | 2.84 | +10% | +14.0 | 0.14 | 0.44 | 0.75 |
| 41 | is_enemy | standing=behind | 3272 | -0.128 | 0.47 | -27% | -11.5 | 0.20 | 0.27 | 0.42 |
| 42 | is_enemy | fmt=2P|standing=behind | 2590 | -0.132 | 0.47 | -28% | -10.7 | 0.21 | 0.27 | 0.41 |
| 43 | garrison | standing=behind | 3272 | +19.946 | 27.11 | +74% | +11.1 | 0.19 | 0.59 | 0.97 |
| 44 | prod | fmt=4P | 1270 | +0.487 | 2.62 | +19% | +9.3 | 0.26 | 0.51 | 0.80 |
| 45 | prod | standing=behind | 3272 | +0.362 | 2.75 | +13% | +10.8 | 0.19 | 0.45 | 0.74 |
| 46 | contested | fmt=2P|standing=behind | 2590 | +0.076 | 0.05 | +160% | +10.2 | 0.20 | 0.11 | 0.15 |
| 47 | contested | standing=behind | 3272 | +0.074 | 0.06 | +125% | +10.5 | 0.18 | 0.12 | 0.17 |
| 48 | prod | contested=uncontested | 8477 | +0.264 | 2.86 | +9% | +12.1 | 0.13 | 0.43 | 0.75 |
| 49 | prod | fmt=2P|phase=mid | 6135 | +0.292 | 2.81 | +10% | +11.4 | 0.15 | 0.44 | 0.75 |
| 50 | contested | all=all | 9508 | +0.047 | 0.06 | +73% | +12.0 | 0.12 | 0.10 | 0.15 |
| 51 | prod | fmt=4P|phase=end | 75 | +1.027 | 2.44 | +42% | +5.0 | 0.58 | 0.61 | 0.79 |
| 52 | contested | fmt=2P | 8238 | +0.047 | 0.06 | +82% | +11.6 | 0.13 | 0.09 | 0.14 |
| 53 | contested | fmt=2P|contested=uncontested | 8238 | +0.047 | 0.06 | +82% | +11.6 | 0.13 | 0.09 | 0.14 |
| 54 | prod | fmt=2P | 8238 | +0.257 | 2.88 | +9% | +11.6 | 0.13 | 0.43 | 0.74 |
| 55 | prod | fmt=2P|contested=uncontested | 8238 | +0.257 | 2.88 | +9% | +11.6 | 0.13 | 0.43 | 0.74 |
| 56 | prod | contested=contested | 1031 | +0.484 | 2.67 | +18% | +8.2 | 0.26 | 0.51 | 0.80 |
| 57 | prod | fmt=4P|contested=contested | 1031 | +0.484 | 2.67 | +18% | +8.2 | 0.26 | 0.51 | 0.80 |
| 58 | contested | contested=uncontested | 8477 | +0.046 | 0.06 | +80% | +11.5 | 0.13 | 0.09 | 0.14 |
| 59 | prod | fmt=4P|phase=mid | 993 | +0.471 | 2.62 | +18% | +8.1 | 0.26 | 0.51 | 0.79 |
| 60 | garrison | fmt=2P|standing=behind | 2590 | +20.562 | 26.39 | +78% | +9.5 | 0.19 | 0.59 | 0.97 |
