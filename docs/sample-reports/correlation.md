# Sim-to-real correlation

* Ranking agreement (Spearman over policies): **0.66**
* Outcome agreement on 100 matched policy/scenario pairs: **68.0%**
* Share of pairs where sim is more optimistic than real: 32.0%

| Policy | Sim | Real | Gap (sim − real) |
|---|---|---|---|
| ckpt_a | 58.3% [45.7%–69.9%] n=60 | 55.0% [34.2%–74.2%] n=20 | 3.3% |
| ckpt_b | 75.0% [62.8%–84.2%] n=60 | 50.0% [29.9%–70.1%] n=20 | 25.0% |
| ckpt_c | 88.3% [77.8%–94.2%] n=60 | 60.0% [38.7%–78.1%] n=20 | 28.3% |
| ckpt_d | 85.0% [73.9%–91.9%] n=60 | 75.0% [53.1%–88.8%] n=20 | 10.0% |
| ckpt_e | 88.3% [77.8%–94.2%] n=60 | 75.0% [53.1%–88.8%] n=20 | 13.3% |

## Scenarios where sim and real disagree most

Candidates for better assets, physics tuning, or domain randomization.

| Scenario | Mean abs disagreement |
|---|---|
| pick_place_12 | 0.73 |
| pick_place_02 | 0.53 |
| pick_place_13 | 0.47 |
| pick_place_00 | 0.47 |
| pick_place_15 | 0.47 |
| pick_place_09 | 0.40 |
| pick_place_16 | 0.40 |
| pick_place_04 | 0.40 |
| pick_place_10 | 0.40 |
| pick_place_17 | 0.40 |
