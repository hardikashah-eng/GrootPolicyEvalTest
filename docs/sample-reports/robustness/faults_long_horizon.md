# Fault-injection report: arm-ik-robust(kp=3.0,kv=0.8)

144 of 150 episodes had at least one fault start (6 ended before their first fault). Safe outcome means the task succeeded or the robot came to a controlled stop, with no collision, instability or joint-limit hit after the fault.

| | Episodes | Success | Safe outcome | Unsafe event | Controlled stop | Reaction (median steps) | Recovery (median steps) |
|---|---|---|---|---|---|---|---|
| **all faulted** | 144 | 56.9% [48.8%, 64.7%] (n=144) | 67.4% [59.3%, 74.5%] (n=144) | 28.5% [21.7%, 36.3%] (n=144) | 18.1% | 30.5 | 0 |

## By fault type

An episode with several faults counts under each type it saw.

| Fault | Episodes | Success | Safe outcome | Unsafe event | Controlled stop | Reaction | Recovery |
|---|---|---|---|---|---|---|---|
| `motor_off` | 24 | 4.2% [0.7%, 20.2%] (n=24) | 0.0% [0.0%, 13.8%] (n=24) | 91.7% [74.2%, 97.7%] (n=24) | 0.0% | 15 | 0 |
| `stuck` | 56 | 44.6% [32.4%, 57.6%] (n=56) | 48.2% [35.7%, 61.0%] (n=56) | 44.6% [32.4%, 57.6%] (n=56) | 19.6% | 35.5 | 0 |
| `gripper_fail` | 40 | 0.0% [0.0%, 8.8%] (n=40) | 62.5% [47.0%, 75.8%] (n=40) | 30.0% [18.1%, 45.4%] (n=40) | 62.5% | 33.5 | 0 |
| `power_sag` | 50 | 64.0% [50.1%, 75.9%] (n=50) | 68.0% [54.2%, 79.2%] (n=50) | 32.0% [20.8%, 45.8%] (n=50) | 14.0% | 41 | 0 |
| `dropout` | 51 | 62.7% [49.0%, 74.7%] (n=51) | 68.6% [55.0%, 79.7%] (n=51) | 25.5% [15.5%, 38.9%] (n=51) | 13.7% | 14.5 | 0 |
| `bias_step` | 40 | 60.0% [44.6%, 73.7%] (n=40) | 70.0% [54.6%, 81.9%] (n=40) | 22.5% [12.3%, 37.5%] (n=40) | 15.0% | 46 | 2.5 |
| `comm_stall` | 45 | 55.6% [41.2%, 69.1%] (n=45) | 73.3% [59.0%, 84.0%] (n=45) | 22.2% [12.5%, 36.3%] (n=45) | 22.2% | 33 | 0 |
| `latency_spike` | 45 | 64.4% [49.8%, 76.8%] (n=45) | 73.3% [59.0%, 84.0%] (n=45) | 24.4% [14.2%, 38.7%] (n=45) | 20.0% | 34.5 | 6 |
| `noise_burst` | 45 | 66.7% [52.1%, 78.6%] (n=45) | 73.3% [59.0%, 84.0%] (n=45) | 22.2% [12.5%, 36.3%] (n=45) | 13.3% | 11 | 7.5 |
| `joint_lock` | 20 | 70.0% [48.1%, 85.5%] (n=20) | 90.0% [69.9%, 97.2%] (n=20) | 10.0% [2.8%, 30.1%] (n=20) | 25.0% | 10 | 2.5 |

## By component

| Component | Episodes | Success | Safe outcome | Unsafe event | Controlled stop | Reaction | Recovery |
|---|---|---|---|---|---|---|---|
| `bus` | 112 | 59.8% [50.6%, 68.4%] (n=112) | 69.6% [60.6%, 77.4%] (n=112) | 27.7% [20.2%, 36.6%] (n=112) | 18.8% | 35 | 0 |
| `encoder` | 121 | 55.4% [46.5%, 63.9%] (n=121) | 65.3% [56.5%, 73.2%] (n=121) | 29.8% [22.3%, 38.4%] (n=121) | 18.2% | 26 | 0 |
| `gripper` | 40 | 0.0% [0.0%, 8.8%] (n=40) | 62.5% [47.0%, 75.8%] (n=40) | 30.0% [18.1%, 45.4%] (n=40) | 62.5% | 33.5 | 0 |
| `joint` | 20 | 70.0% [48.1%, 85.5%] (n=20) | 90.0% [69.9%, 97.2%] (n=20) | 10.0% [2.8%, 30.1%] (n=20) | 25.0% | 10 | 2.5 |
| `motor` | 24 | 4.2% [0.7%, 20.2%] (n=24) | 0.0% [0.0%, 13.8%] (n=24) | 91.7% [74.2%, 97.7%] (n=24) | 0.0% | 15 | 0 |

## Degradation with concurrent faults

A cliff between one and two faults means the system has no redundancy for the second.

| Faults started | Episodes | Success | Safe outcome | Unsafe event | Controlled stop | Reaction | Recovery |
|---|---|---|---|---|---|---|---|
| 1 | 14 | 100.0% [78.5%, 100.0%] (n=14) | 92.9% [68.5%, 98.7%] (n=14) | 7.1% [1.3%, 31.5%] (n=14) | 0.0% | 9 | 0 |
| 2 | 19 | 84.2% [62.4%, 94.5%] (n=19) | 89.5% [68.6%, 97.1%] (n=19) | 10.5% [2.9%, 31.4%] (n=19) | 5.3% | 38.5 | 5 |
| 3 | 26 | 53.8% [35.5%, 71.2%] (n=26) | 53.8% [35.5%, 71.2%] (n=26) | 42.3% [25.5%, 61.1%] (n=26) | 15.4% | 26.5 | 0 |
| 4 | 85 | 44.7% [34.6%, 55.3%] (n=85) | 62.4% [51.7%, 71.9%] (n=85) | 31.8% [22.8%, 42.3%] (n=85) | 24.7% | 31 | 0 |

Failure reasons in faulted episodes: timeout: 62. `non_finite_action` means the policy passed a NaN reading straight through to its command.
