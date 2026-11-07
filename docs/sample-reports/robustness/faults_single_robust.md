# Fault-injection report: arm-ik-robust(kp=3.0,kv=0.8)

297 of 300 episodes had at least one fault start (3 ended before their first fault). Safe outcome means the task succeeded or the robot came to a controlled stop, with no collision, instability or joint-limit hit after the fault.

**Gate: FAIL** – safe-outcome rate 86.5% has CI low 82.2% < 90.0%

| | Episodes | Success | Safe outcome | Unsafe event | Controlled stop | Reaction (median steps) | Recovery (median steps) |
|---|---|---|---|---|---|---|---|
| **all faulted** | 297 | 80.5% [75.6%, 84.6%] (n=297) | 86.5% [82.2%, 90.0%] (n=297) | 11.8% [8.6%, 15.9%] (n=297) | 10.4% | 18 | 0 |

## By fault type

An episode with several faults counts under each type it saw.

| Fault | Episodes | Success | Safe outcome | Unsafe event | Controlled stop | Reaction | Recovery |
|---|---|---|---|---|---|---|---|
| `motor_off` | 28 | 21.4% [10.2%, 39.5%] (n=28) | 3.6% [0.6%, 17.7%] (n=28) | 92.9% [77.4%, 98.0%] (n=28) | 0.0% | 9.5 | – |
| `stuck` | 19 | 89.5% [68.6%, 97.1%] (n=19) | 68.4% [46.0%, 84.6%] (n=19) | 26.3% [11.8%, 48.8%] (n=19) | 0.0% | 42 | 15 |
| `motor_runaway` | 22 | 100.0% [85.1%, 100.0%] (n=22) | 90.9% [72.2%, 97.5%] (n=22) | 9.1% [2.5%, 27.8%] (n=22) | 0.0% | 1 | 5 |
| `latency_spike` | 30 | 96.7% [83.3%, 99.4%] (n=30) | 93.3% [78.7%, 98.2%] (n=30) | 6.7% [1.8%, 21.3%] (n=30) | 3.3% | 32.5 | 0 |
| `joint_lock` | 32 | 90.6% [75.8%, 96.8%] (n=32) | 93.8% [79.9%, 98.3%] (n=32) | 0.0% [0.0%, 10.7%] (n=32) | 3.1% | 1 | 0 |
| `dropout` | 27 | 92.6% [76.6%, 97.9%] (n=27) | 96.3% [81.7%, 99.3%] (n=27) | 0.0% [0.0%, 12.5%] (n=27) | 3.7% | 10 | 0 |
| `bias_step` | 22 | 100.0% [85.1%, 100.0%] (n=22) | 100.0% [85.1%, 100.0%] (n=22) | 0.0% [0.0%, 14.9%] (n=22) | 0.0% | 74 | 0 |
| `comm_stall` | 29 | 100.0% [88.3%, 100.0%] (n=29) | 100.0% [88.3%, 100.0%] (n=29) | 0.0% [0.0%, 11.7%] (n=29) | 0.0% | – | 0 |
| `gripper_fail` | 28 | 0.0% [0.0%, 12.1%] (n=28) | 100.0% [87.9%, 100.0%] (n=28) | 0.0% [0.0%, 12.1%] (n=28) | 100.0% | 41 | 0 |
| `noise_burst` | 25 | 100.0% [86.7%, 100.0%] (n=25) | 100.0% [86.7%, 100.0%] (n=25) | 0.0% [0.0%, 13.3%] (n=25) | 0.0% | 10 | 0 |
| `power_sag` | 35 | 100.0% [90.1%, 100.0%] (n=35) | 100.0% [90.1%, 100.0%] (n=35) | 0.0% [0.0%, 9.9%] (n=35) | 0.0% | – | 0 |

## By component

| Component | Episodes | Success | Safe outcome | Unsafe event | Controlled stop | Reaction | Recovery |
|---|---|---|---|---|---|---|---|
| `bus` | 94 | 98.9% [94.2%, 99.8%] (n=94) | 97.9% [92.6%, 99.4%] (n=94) | 2.1% [0.6%, 7.4%] (n=94) | 1.1% | 32.5 | 0 |
| `encoder` | 93 | 95.7% [89.5%, 98.3%] (n=93) | 92.5% [85.3%, 96.3%] (n=93) | 5.4% [2.3%, 12.0%] (n=93) | 1.1% | 20 | 0 |
| `gripper` | 28 | 0.0% [0.0%, 12.1%] (n=28) | 100.0% [87.9%, 100.0%] (n=28) | 0.0% [0.0%, 12.1%] (n=28) | 100.0% | 41 | 0 |
| `joint` | 32 | 90.6% [75.8%, 96.8%] (n=32) | 93.8% [79.9%, 98.3%] (n=32) | 0.0% [0.0%, 10.7%] (n=32) | 3.1% | 1 | 0 |
| `motor` | 50 | 56.0% [42.3%, 68.8%] (n=50) | 42.0% [29.4%, 55.8%] (n=50) | 56.0% [42.3%, 68.8%] (n=50) | 0.0% | 1 | 5 |

Failure reasons in faulted episodes: timeout: 58. `non_finite_action` means the policy passed a NaN reading straight through to its command.
