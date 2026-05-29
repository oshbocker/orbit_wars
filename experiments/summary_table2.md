| Experiment | Updates | Avg Reward (last 50) | Final Loss | Final Value Loss | Avg Entropy | vs Apex | vs Random | Avg dt |
|---|---|---|---|---|---|---|---|---|
| baseline | 200 | -0.356 | 0.1856 | 0.2677 | 9.05 | 0% | 100% | 11.7s |
| r1_ent_anneal | 200 | -0.226 | 0.0308 | 0.0811 | 4.27 | 0% | 100% | 11.5s |
| r2_high_gamma | 200 | -0.028 | 0.0454 | 0.0043 | 11.98 | 0% | 100% | 11.2s |
| r3_strong_value | 200 | -0.023 | -0.0193 | 0.8764 | 35.00 | 0% | 100% | 12.7s |
| r4_fast_selfplay | 200 | -0.029 | -0.0864 | 0.0220 | 20.78 | 0% | 100% | 11.1s |
| r5_no_prod_bonus | 200 | -0.082 | -0.1649 | 0.0307 | 8.94 | 0% | 100% | 11.8s |