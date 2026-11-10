# Aging report: arm-ik(kp=3.0,kv=0.8)

Success goes from 100.0% on a `new` robot to 78.5% at `end_of_life` (10000 h), on the same 200 scenarios and units.
**Robustness margin:** the success lower bound first drops below 90.0% at `year_3` (8760 h); last passing profile: `year_2`.

## Degradation curve

Paired test: McNemar on the same scenario and unit across ages (lost = succeeded on the reference robot, failed on this one).

| Profile | Age (h) | Success [95% CI] | Lost / gained vs ref | p | Mean steps | Final dist | Torque-sat steps | Limit hits |
|---|---|---|---|---|---|---|---|---|
| new | 0 | 100.0% [98.1%, 100.0%] (n=200) | reference | – | 74.4 | 0.0219 | 2.71 | 0 |
| week_1 | 56 | 100.0% [98.1%, 100.0%] (n=200) | 0 / 0 | 1 | 74.3 | 0.0219 | 2.63 | 0 |
| month_6 | 1456 | 100.0% [98.1%, 100.0%] (n=200) | 0 / 0 | 1 | 74.3 | 0.0218 | 2.25 | 0 |
| year_1 | 2920 | 100.0% [98.1%, 100.0%] (n=200) | 0 / 0 | 1 | 75.5 | 0.0213 | 2.48 | 0 |
| year_2 | 5840 | 97.5% [94.3%, 98.9%] (n=200) | 5 / 0 | 0.0625 | 91.1 | 0.0211 | 3.28 | 0 |
| year_3 | 8760 | 87.0% [81.6%, 91.0%] (n=200) | 26 / 0 | **2.98e-08** | 132 | 0.026 | 9.2 | 0.03 |
| end_of_life | 10000 | 78.5% [72.3%, 83.6%] (n=200) | 43 / 0 | **2.27e-13** | 155 | 0.0317 | 16.8 | 0.095 |

Failure reasons at `end_of_life`: timeout: 43.

## Which wear drives failure at `end_of_life`

Units at the same age differ (log-normal spread), so per-unit wear values can be ranked like any randomized parameter. Only gaps whose intervals do not overlap are significant.

| Wear parameter | Gap | Significant | Worst bucket |
|---|---|---|---|
| `wear.joint[1].friction_scale` | 26.0% | **yes** | 2.15–3.14 |
| `wear.motor[2].gain_scale` | 22.0% | no | 0.755–0.805 |
| `wear.encoder[0].bias_rad` | 20.0% | no | 0.031–0.0786 |
| `wear.motor[2].torque_limit_scale` | 16.0% | no | 0.755–0.793 |
| `wear.encoder[1].bias_rad` | 16.0% | no | 0.0146–0.0349 |
| `wear.joint[0].stiction` | 14.0% | no | 0.188–0.507 |
| `wear.encoder[2].bias_rad` | 14.0% | no | -0.142–-0.0261 |
| `wear.motor[0].torque_limit_scale` | 12.0% | no | 0.133–0.534 |
| `wear.joint[1].backlash_rad` | 12.0% | no | 0.0151–0.0304 |
| `wear.motor[0].gain_scale` | 10.0% | no | 0.356–0.634 |
| `wear.joint[0].friction_scale` | 10.0% | no | 3.92–4.77 |
| `wear.joint[2].backlash_rad` | 10.0% | no | 0.00581–0.0196 |

## Loss of a degree of freedom

One joint at a time, on an otherwise new robot. `locked`: seized at its current angle; `limp`: motor off, joint swings under gravity; `limited`: 30% of its range. The policy still sends commands to that joint, as on a real robot.

| Joint | locked | limp | limited |
|---|---|---|---|
| (none) | 100.0% [91.2%, 100.0%] (n=40) | 100.0% [91.2%, 100.0%] (n=40) | 100.0% [91.2%, 100.0%] (n=40) |
| joint[0] | 47.5% [32.9%, 62.5%] (n=40) | 0.0% [0.0%, 8.8%] (n=40) | 77.5% [62.5%, 87.7%] (n=40) |
| joint[1] | 87.5% [73.9%, 94.5%] (n=40) | 5.0% [1.4%, 16.5%] (n=40) | 100.0% [91.2%, 100.0%] (n=40) |
| joint[2] | 77.5% [62.5%, 87.7%] (n=40) | 10.0% [4.0%, 23.1%] (n=40) | 77.5% [62.5%, 87.7%] (n=40) |
