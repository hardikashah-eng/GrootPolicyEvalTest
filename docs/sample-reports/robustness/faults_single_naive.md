# Fault-injection report: arm-ik(kp=3.0,kv=0.8)

297 of 300 episodes had at least one fault start (3 ended before their first fault). Safe outcome means the task succeeded or the robot came to a controlled stop, with no collision, instability or joint-limit hit after the fault.

**Gate: FAIL** – safe-outcome rate 77.8% has CI low 72.7% < 90.0%

| | Episodes | Success | Safe outcome | Unsafe event | Controlled stop | Reaction (median steps) | Recovery (median steps) |
|---|---|---|---|---|---|---|---|
| **all faulted** | 297 | 72.1% [66.7%, 76.9%] (n=297) | 77.8% [72.7%, 82.1%] (n=297) | 21.5% [17.3%, 26.6%] (n=297) | 10.1% | 20.5 | 0 |

## By fault type

An episode with several faults counts under each type it saw.

| Fault | Episodes | Success | Safe outcome | Unsafe event | Controlled stop | Reaction | Recovery |
|---|---|---|---|---|---|---|---|
| `dropout` | 27 | 0.0% [0.0%, 12.5%] (n=27) | 0.0% [0.0%, 12.5%] (n=27) | 100.0% [87.5%, 100.0%] (n=27) | 0.0% | – | – |
| `motor_off` | 28 | 21.4% [10.2%, 39.5%] (n=28) | 3.6% [0.6%, 17.7%] (n=28) | 92.9% [77.4%, 98.0%] (n=28) | 0.0% | 9.5 | – |
| `stuck` | 19 | 78.9% [56.7%, 91.5%] (n=19) | 57.9% [36.3%, 76.9%] (n=19) | 36.8% [19.1%, 59.0%] (n=19) | 0.0% | 35.5 | 22 |
| `motor_runaway` | 22 | 100.0% [85.1%, 100.0%] (n=22) | 90.9% [72.2%, 97.5%] (n=22) | 9.1% [2.5%, 27.8%] (n=22) | 0.0% | 1 | 5 |
| `latency_spike` | 30 | 96.7% [83.3%, 99.4%] (n=30) | 93.3% [78.7%, 98.2%] (n=30) | 6.7% [1.8%, 21.3%] (n=30) | 3.3% | 32.5 | 0 |
| `bias_step` | 22 | 100.0% [85.1%, 100.0%] (n=22) | 100.0% [85.1%, 100.0%] (n=22) | 0.0% [0.0%, 14.9%] (n=22) | 0.0% | 74 | 0 |
| `comm_stall` | 29 | 100.0% [88.3%, 100.0%] (n=29) | 100.0% [88.3%, 100.0%] (n=29) | 0.0% [0.0%, 11.7%] (n=29) | 0.0% | – | 0 |
| `gripper_fail` | 28 | 0.0% [0.0%, 12.1%] (n=28) | 100.0% [87.9%, 100.0%] (n=28) | 0.0% [0.0%, 12.1%] (n=28) | 100.0% | 41 | 0 |
| `joint_lock` | 32 | 96.9% [84.3%, 99.4%] (n=32) | 100.0% [89.3%, 100.0%] (n=32) | 0.0% [0.0%, 10.7%] (n=32) | 3.1% | 1 | 0 |
| `noise_burst` | 25 | 100.0% [86.7%, 100.0%] (n=25) | 100.0% [86.7%, 100.0%] (n=25) | 0.0% [0.0%, 13.3%] (n=25) | 0.0% | 10 | 0 |
| `power_sag` | 35 | 100.0% [90.1%, 100.0%] (n=35) | 100.0% [90.1%, 100.0%] (n=35) | 0.0% [0.0%, 9.9%] (n=35) | 0.0% | – | 0 |

## By component

| Component | Episodes | Success | Safe outcome | Unsafe event | Controlled stop | Reaction | Recovery |
|---|---|---|---|---|---|---|---|
| `bus` | 94 | 98.9% [94.2%, 99.8%] (n=94) | 97.9% [92.6%, 99.4%] (n=94) | 2.1% [0.6%, 7.4%] (n=94) | 1.1% | 32.5 | 0 |
| `encoder` | 93 | 66.7% [56.6%, 75.4%] (n=93) | 62.4% [52.2%, 71.5%] (n=93) | 36.6% [27.5%, 46.7%] (n=93) | 0.0% | 28 | 0 |
| `gripper` | 28 | 0.0% [0.0%, 12.1%] (n=28) | 100.0% [87.9%, 100.0%] (n=28) | 0.0% [0.0%, 12.1%] (n=28) | 100.0% | 41 | 0 |
| `joint` | 32 | 96.9% [84.3%, 99.4%] (n=32) | 100.0% [89.3%, 100.0%] (n=32) | 0.0% [0.0%, 10.7%] (n=32) | 3.1% | 1 | 0 |
| `motor` | 50 | 56.0% [42.3%, 68.8%] (n=50) | 42.0% [29.4%, 55.8%] (n=50) | 56.0% [42.3%, 68.8%] (n=50) | 0.0% | 1 | 5 |

Failure reasons in faulted episodes: timeout: 56, non_finite_action: 27. `non_finite_action` means the policy passed a NaN reading straight through to its command.
