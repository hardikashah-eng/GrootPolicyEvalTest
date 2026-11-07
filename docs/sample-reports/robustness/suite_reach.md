# Skill suite: reach_grasp – arm-ik(kp=3.0,kv=0.8)

Overall 89.7% [86.8%, 92.0%] (n=524) across oat, corners, lhs.
All conditions nominal: 100.0% [75.7%, 100.0%] (n=12).

**Gate: FAIL** – corner target_radius=min,payload_kg=max,friction=min,action_delay=max: 0.0%, CI high 24.3% < 50%

## One condition at a time

Each condition at its min and max, every other condition nominal. Significant = the worst and best levels' intervals do not overlap.

| Condition | Min | Nominal | Max | Gap | Significant |
|---|---|---|---|---|---|
| `action_delay` | = nominal | 0: 100.0% [75.7%, 100.0%] (n=12) | 2: 83.3% [55.2%, 95.3%] (n=12) | 16.7% | no |
| `target_radius` | 0.35: 100.0% [75.7%, 100.0%] (n=12) | 0.6: 100.0% [75.7%, 100.0%] (n=12) | 0.82: 100.0% [75.7%, 100.0%] (n=12) | 0.0% | no |
| `target_angle` | 0.3: 100.0% [75.7%, 100.0%] (n=12) | 1.57: 100.0% [75.7%, 100.0%] (n=12) | 2.85: 100.0% [75.7%, 100.0%] (n=12) | 0.0% | no |
| `payload_kg` | 0.0: 100.0% [75.7%, 100.0%] (n=12) | 0.1: 100.0% [75.7%, 100.0%] (n=12) | 0.6: 100.0% [75.7%, 100.0%] (n=12) | 0.0% | no |
| `friction` | 0.6: 100.0% [75.7%, 100.0%] (n=12) | 1.0: 100.0% [75.7%, 100.0%] (n=12) | 2.5: 100.0% [75.7%, 100.0%] (n=12) | 0.0% | no |
| `sensor_noise` | = nominal | 0.0: 100.0% [75.7%, 100.0%] (n=12) | 0.01: 100.0% [75.7%, 100.0%] (n=12) | 0.0% | no |

## Worst two-way combinations

Pooled over every corner that contains the pair; the other conditions vary over min and max.

| Combination | Success |
|---|---|
| friction=min, action_delay=max | 39.6% [27.0%, 53.7%] (n=48) |
| target_radius=min, friction=min | 60.4% [46.3%, 73.0%] (n=48) |
| target_radius=min, action_delay=max | 60.4% [46.3%, 73.0%] (n=48) |
| payload_kg=min, action_delay=max | 64.6% [50.4%, 76.6%] (n=48) |
| payload_kg=max, action_delay=max | 64.6% [50.4%, 76.6%] (n=48) |
| target_radius=max, action_delay=max | 68.8% [54.7%, 80.1%] (n=48) |
| payload_kg=min, friction=min | 68.8% [54.7%, 80.1%] (n=48) |
| payload_kg=max, friction=min | 70.8% [56.8%, 81.8%] (n=48) |

## Worst corners

| Corner (all conditions at min or max) | Success |
|---|---|
| target_radius=min,payload_kg=max,friction=min,action_delay=max | 0.0% [0.0%, 24.3%] (n=12) |
| target_radius=max,payload_kg=min,friction=min,action_delay=max | 33.3% [13.8%, 60.9%] (n=12) |
| target_radius=min,payload_kg=min,friction=min,action_delay=max | 41.7% [19.3%, 68.0%] (n=12) |
| target_radius=max,payload_kg=max,friction=max,action_delay=max | 75.0% [46.8%, 91.1%] (n=12) |
| target_radius=max,payload_kg=min,friction=max,action_delay=max | 83.3% [55.2%, 95.3%] (n=12) |
| target_radius=max,payload_kg=max,friction=min,action_delay=max | 83.3% [55.2%, 95.3%] (n=12) |
| target_radius=min,payload_kg=min,friction=min,action_delay=min | 100.0% [75.7%, 100.0%] (n=12) |
| target_radius=min,payload_kg=min,friction=max,action_delay=min | 100.0% [75.7%, 100.0%] (n=12) |

## Space-filling sample (LHS): 91.0% [86.2%, 94.2%] (n=200)

| Condition | Gap | Significant | Worst bucket |
|---|---|---|---|
| `action_delay` | 36.0% | **yes** | 2 |
| `payload_kg` | 10.0% | no | 0.00298–0.151 |
| `friction` | 10.0% | no | 1.07–1.55 |
| `sensor_noise` | 10.0% | no | 4.52e-05–0.00251 |
| `target_radius` | 8.0% | no | 0.584–0.702 |
| `target_angle` | 8.0% | no | 0.939–1.58 |

## Skill metrics

| Metric | Mean | Min | Max |
|---|---|---|---|
| final_distance | 0.02257 | 0.004261 | 0.0746 |
| path_length | 1.528 | 0.01421 | 7.106 |
| torque_saturation_steps | 89.37 | 0 | 473 |
