# Robustness demo (SYNTHETIC: Arm3 simulator, placeholder wear profiles)

## Aging

| Profile | Age (h) | Success [95% CI] | Lost vs new (McNemar p) |
|---|---|---|---|
| new | 0 | 100.0% [98.1%, 100.0%] | reference |
| week_1 | 56 | 100.0% [98.1%, 100.0%] | 0 (p=1) |
| month_6 | 1456 | 100.0% [98.1%, 100.0%] | 0 (p=1) |
| year_1 | 2920 | 100.0% [98.1%, 100.0%] | 0 (p=1) |
| year_2 | 5840 | 97.5% [94.3%, 98.9%] | 5 (p=0.062) |
| year_3 | 8760 | 87.0% [81.6%, 91.0%] | 26 (p=3e-08) |
| end_of_life | 10000 | 78.5% [72.3%, 83.6%] | 43 (p=2.3e-13) |

Robustness margin at 90%: first failing profile `year_3`, last passing `year_2`. Gate `--min-success-at year_1 0.9`: exit 0.
Significant wear drivers at end of life: `wear.joint[1].friction_scale`.

## Single faults: naive vs robust controller

| Controller | Success | Safe outcome [95% CI] | Unsafe | NaN commands |
|---|---|---|---|---|
| arm-ik | 72.1% | 77.8% [72.7%, 82.1%] | 21.5% | 27 |
| arm-ik-robust | 80.5% | 86.5% [82.2%, 90.0%] | 11.8% | 0 |

Worst fault types for the robust controller: `motor_off` (safe 3.6%), `stuck` (safe 68.4%), `motor_runaway` (safe 90.9%).

## Long horizon, several random faults, 1-year-old robot

| Faults started | Episodes | Success | Safe outcome |
|---|---|---|---|
| 1 | 14 | 100.0% | 92.9% |
| 2 | 19 | 84.2% | 89.5% |
| 3 | 26 | 53.8% | 53.8% |
| 4 | 85 | 44.7% | 62.4% |

## Skill suite (reach and grasp)

Overall 89.7%; all-nominal centre 100.0%. One-at-a-time significant conditions: none. Worst two-way combination: friction=min, action_delay=max at 39.6%. Gate `--min-corner 0.5`: exit 1 (1 corners significantly below).

## Data-fidelity gates

| Run | Exit | G0 | G1 | G2 | G3 | G4 | G5 | G6 |
|---|---|---|---|---|---|---|---|---|
| fidelity_units_bug | 1 | warn | fail | skip | skip | skip | skip | skip |
| fidelity_worn_robot | 1 | pass | pass | pass | fail | fail | fail | skip |
| fidelity_after_sysid | 1 | pass | pass | pass | fail | fail | fail | skip |
| fidelity_self_check | 0 | pass | pass | pass | pass | pass | pass | skip |

G5 replay error at 10 steps: self-check 0.000, worn robot vs nominal sim 1.299, after fitting friction 0.486 (best friction 8.00, at the edge of the searched range, which G5 flags). One global friction parameter is absorbing per-joint wear (backlash, gain loss, encoder bias) it cannot represent: the sim needs per-joint wear parameters, which is what `--wear` provides.
