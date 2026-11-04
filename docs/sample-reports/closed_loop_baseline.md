# Closed-loop evaluation: `pd(kp=2.0,kd=1.2)` on `reach2d`

**Success rate 62.3%** (187/300 episodes), 95% CI 56.7% – 67.6%.

| Metric | Value |
|---|---|
| Episodes | 300 |
| Mean steps | 39.7 |
| Policy latency p50 / p95 / p99 (ms) | 0.01 / 0.01 / 0.02 |

## Failure reasons

| Reason | Episodes |
|---|---|
| collision | 110 |
| timeout | 3 |

## What drives failure

Success rate by bucket of each randomized parameter. **Significant** means the best and worst buckets' confidence intervals do not overlap.

| Parameter | Gap | Significant | Worst bucket | Buckets (success rate, n) |
|---|---|---|---|---|
| obstacle | 96.6% | **yes** | True | False: 98.4% (n=188); True: 1.8% (n=112) |
| action_delay | 15.6% | no | 4 | 0: 71.0% (n=69); 1: 64.9% (n=77); 2: 58.8% (n=80); 4: 55.4% (n=74) |
| actuator_gain | 9.3% | no | 0.904–1.06 | 0.602–0.76: 62.7% (n=75); 0.76–0.904: 65.3% (n=75); 0.904–1.06: 56.0% (n=75); 1.06–1.2: 65.3% (n=75) |
| sensor_noise | 9.3% | no | 0.0295–0.0397 | 0.000207–0.0108: 62.7% (n=75); 0.0108–0.02: 62.7% (n=75); 0.02–0.0295: 66.7% (n=75); 0.0295–0.0397: 57.3% (n=75) |
| friction | 8.0% | no | 0.947–1.2 | 0.204–0.427: 65.3% (n=75); 0.427–0.673: 62.7% (n=75); 0.673–0.947: 64.0% (n=75); 0.947–1.2: 57.3% (n=75) |
| mass | 8.0% | no | 1.25–1.6 | 0.523–0.845: 61.3% (n=75); 0.845–1.25: 62.7% (n=75); 1.25–1.6: 58.7% (n=75); 1.6–2: 66.7% (n=75) |

## Worst cases

| Scenario | Seed | Reason | Steps | Params |
|---|---|---|---|---|
| `559ffbb12fa5` | 264804365 | collision | 10 | friction=0.340, mass=0.592, actuator_gain=1.052, sensor_noise=0.023, action_delay=2, obstacle=True |
| `7e6323f783d8` | 1885894066 | collision | 19 | friction=0.451, mass=1.709, actuator_gain=1.006, sensor_noise=0.029, action_delay=2, obstacle=True |
| `954d8d2eadeb` | 1910048758 | collision | 13 | friction=0.964, mass=0.589, actuator_gain=1.022, sensor_noise=0.037, action_delay=4, obstacle=True |
| `b56d90c941d6` | 1585174891 | collision | 15 | friction=0.341, mass=0.747, actuator_gain=0.830, sensor_noise=0.019, action_delay=4, obstacle=True |
| `cf8e7c7be0a0` | 451549389 | collision | 15 | friction=0.982, mass=1.203, actuator_gain=1.138, sensor_noise=0.015, action_delay=2, obstacle=True |
| `55059bc673b9` | 1393211573 | collision | 9 | friction=0.978, mass=0.525, actuator_gain=1.165, sensor_noise=0.010, action_delay=1, obstacle=True |
| `9014cd80d430` | 149061184 | collision | 22 | friction=1.089, mass=1.710, actuator_gain=0.707, sensor_noise=0.039, action_delay=0, obstacle=True |
| `352b28e28dbd` | 1329597501 | collision | 17 | friction=0.339, mass=1.948, actuator_gain=0.999, sensor_noise=0.032, action_delay=2, obstacle=True |
| `ed5759454fd0` | 1906251987 | collision | 19 | friction=1.042, mass=1.192, actuator_gain=0.758, sensor_noise=0.019, action_delay=1, obstacle=True |
| `d22e5b53b21d` | 19296717 | collision | 15 | friction=1.051, mass=1.437, actuator_gain=1.194, sensor_noise=0.003, action_delay=1, obstacle=True |

Replay any of these exactly with `groot-eval regress` using the exported suite.
