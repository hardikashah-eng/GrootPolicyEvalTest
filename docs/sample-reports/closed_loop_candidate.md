# Closed-loop evaluation: `pd(kp=2.0,kd=1.2,avoid=2.0)` on `reach2d`

**Success rate 73.3%** (220/300 episodes), 95% CI 68.1% – 78.0%.

| Metric | Value |
|---|---|
| Episodes | 300 |
| Mean steps | 46.5 |
| Policy latency p50 / p95 / p99 (ms) | 0.01 / 0.02 / 0.03 |

## Failure reasons

| Reason | Episodes |
|---|---|
| collision | 77 |
| timeout | 3 |

## What drives failure

Success rate by bucket of each randomized parameter. **Significant** means the best and worst buckets' confidence intervals do not overlap.

| Parameter | Gap | Significant | Worst bucket | Buckets (success rate, n) |
|---|---|---|---|---|
| obstacle | 67.2% | **yes** | True | False: 98.4% (n=188); True: 31.2% (n=112) |
| action_delay | 36.0% | **yes** | 4 | 0: 92.8% (n=69); 1: 79.2% (n=77); 2: 66.2% (n=80); 4: 56.8% (n=74) |
| actuator_gain | 14.7% | no | 0.904–1.06 | 0.602–0.76: 81.3% (n=75); 0.76–0.904: 70.7% (n=75); 0.904–1.06: 66.7% (n=75); 1.06–1.2: 74.7% (n=75) |
| sensor_noise | 13.3% | no | 0.0295–0.0397 | 0.000207–0.0108: 74.7% (n=75); 0.0108–0.02: 74.7% (n=75); 0.02–0.0295: 78.7% (n=75); 0.0295–0.0397: 65.3% (n=75) |
| mass | 10.7% | no | 0.523–0.845 | 0.523–0.845: 69.3% (n=75); 0.845–1.25: 69.3% (n=75); 1.25–1.6: 74.7% (n=75); 1.6–2: 80.0% (n=75) |
| friction | 9.3% | no | 0.204–0.427 | 0.204–0.427: 69.3% (n=75); 0.427–0.673: 73.3% (n=75); 0.673–0.947: 72.0% (n=75); 0.947–1.2: 78.7% (n=75) |

## Worst cases

| Scenario | Seed | Reason | Steps | Params |
|---|---|---|---|---|
| `7e6323f783d8` | 1885894066 | collision | 19 | friction=0.451, mass=1.709, actuator_gain=1.006, sensor_noise=0.029, action_delay=2, obstacle=True |
| `559ffbb12fa5` | 264804365 | collision | 10 | friction=0.340, mass=0.592, actuator_gain=1.052, sensor_noise=0.023, action_delay=2, obstacle=True |
| `cf8e7c7be0a0` | 451549389 | collision | 15 | friction=0.982, mass=1.203, actuator_gain=1.138, sensor_noise=0.015, action_delay=2, obstacle=True |
| `954d8d2eadeb` | 1910048758 | collision | 13 | friction=0.964, mass=0.589, actuator_gain=1.022, sensor_noise=0.037, action_delay=4, obstacle=True |
| `b56d90c941d6` | 1585174891 | collision | 15 | friction=0.341, mass=0.747, actuator_gain=0.830, sensor_noise=0.019, action_delay=4, obstacle=True |
| `b31c17d1708d` | 514459690 | collision | 24 | friction=1.074, mass=1.493, actuator_gain=0.679, sensor_noise=0.034, action_delay=4, obstacle=True |
| `55059bc673b9` | 1393211573 | collision | 10 | friction=0.978, mass=0.525, actuator_gain=1.165, sensor_noise=0.010, action_delay=1, obstacle=True |
| `d12f85271fb0` | 860148480 | collision | 12 | friction=0.674, mass=0.523, actuator_gain=0.875, sensor_noise=0.007, action_delay=1, obstacle=True |
| `464e21839c6f` | 1490184163 | collision | 14 | friction=0.384, mass=0.956, actuator_gain=0.869, sensor_noise=0.037, action_delay=1, obstacle=True |
| `48540aca34d9` | 1417250615 | collision | 13 | friction=0.401, mass=0.975, actuator_gain=1.024, sensor_noise=0.031, action_delay=0, obstacle=True |

Replay any of these exactly with `groot-eval regress` using the exported suite.
