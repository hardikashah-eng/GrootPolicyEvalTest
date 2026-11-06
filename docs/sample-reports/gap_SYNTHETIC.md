> **SYNTHETIC.** Both sides are the built-in Reach2D simulator: "sim" has friction 0.4-0.8; "real" is a stand-in with friction 0.9-1.2, sensor noise 0.03 and 2 steps of hidden actuation delay. Policy `pd-chunk`, 4 steps per chunk, 20 episodes each. It shows the report format, not a real robot result.

# Sim-to-real gap: `sim` vs `real`

`sim`: 20 episodes, 903 steps (`sim_trace`).  
`real`: 20 episodes, 1242 steps (`real_trace`).

## 1. Contract mismatches (fix these first)

None found: keys, dimensions, units, ranges, control rate and image range are consistent.

## 2. Largest gaps

Ranked by Wasserstein distance in pooled standard deviations (0 = identical, above about 0.5 is a real difference). KS p-values use episodes as the sample size.

| # | Signal | sim mean ± std | real mean ± std | W1/σ | KS | p | Likely cause |
|---|---|---|---|---|---|---|---|
| 1 | state: `velocity[1]` | 0.52 ± 0.388 | 0.371 ± 0.349 | 0.41 | 0.2 | 0.78 | small; distribution shape differs: dynamics (friction, mass, damping, gains) or behaviour |
| 2 | state: `target[1]` | 0.617 ± 0.106 | 0.595 ± 0.122 | 0.36 | 0.25 | 0.49 | small; distribution shape differs: dynamics (friction, mass, damping, gains) or behaviour |
| 3 | state: `velocity[0]` | 0.506 ± 0.381 | 0.379 ± 0.358 | 0.34 | 0.18 | 0.88 | small; distribution shape differs: dynamics (friction, mass, damping, gains) or behaviour |
| 4 | action_change: `action[1]` | 0.0299 ± 0.0568 | 0.0475 ± 0.0597 | 0.31 | 0.34 | 0.16 | small; commands are jerkier in real: latency, chunk boundaries or noisier observations |
| 5 | action_change: `action[0]` | 0.0297 ± 0.0565 | 0.0465 ± 0.0591 | 0.29 | 0.33 | 0.18 | small; commands are jerkier in real: latency, chunk boundaries or noisier observations |
| 6 | state: `target[0]` | 0.621 ± 0.12 | 0.586 ± 0.126 | 0.28 | 0.26 | 0.45 | small; distribution shape differs: dynamics (friction, mass, damping, gains) or behaviour |
| 7 | action: `action[0]` | 0.15 ± 0.366 | 0.211 ± 0.343 | 0.17 | 0.51 | 0.0061 | close |
| 8 | action: `action[1]` | 0.155 ± 0.371 | 0.206 ± 0.341 | 0.17 | 0.52 | 0.0052 | close |
| 9 | state: `position[0]` | 0.195 ± 0.395 | 0.174 ± 0.407 | 0.073 | 0.066 | 1 | close |
| 10 | state: `position[1]` | 0.178 ± 0.401 | 0.197 ± 0.398 | 0.069 | 0.044 | 1 | close |

## 3. Timing

| Measure | sim | real |
|---|---|---|
| Control rate (Hz) | 20 | 20 |
| Control period from timestamps (ms) | 50 ± 0 (p95 50, max 50) | 50 ± 0 (p95 50, max 50) |
| Host-clock step period (ms) | 0.078 ± 0.081 (p95 0.219, max 0.354) | 0.0823 ± 0.084 (p95 0.227, max 0.366) |
| Inference latency (ms) | 0.0478 ± 0.079 (p95 0.185, max 0.272) | 0.0482 ± 0.081 (p95 0.189, max 0.315) |
| Latency p95 in control steps | 0.0037 | 0.0038 |

In simulation the host-clock period measures compute speed, not a control rate; compare it only between real runs.

* No timing problems detected.

## 4. Action chunking

| Measure | sim | real |
|---|---|---|
| Mean action change at a new chunk | 0.0559 | 0.132 |
| Mean action change inside a chunk | 0.0416 | 0.0525 |
| Ratio (1 = seamless) | 1.34 | 2.52 |

A ratio well above 1 means each new chunk starts from a different place than the last one ended: the policy's prediction of the next state is off (worse when latency is unmodelled).

**Chunk boundaries jump 1.9x more in `real` than in `sim`.** The policy's internal model of what happens during a chunk fits `sim` better: look for actuation delay, latency or dynamics that `sim` lacks.

## 5. Outcomes

| | Success | 95% CI | Failure reasons |
|---|---|---|---|
| sim | 20/20 (100.0%) | 83.9–100.0% | none |
| real | 20/20 (100.0%) | 83.9–100.0% | none |

---
Distribution comparison only: differences can come from behaviour as well as physics. Fix contract mismatches, then check the top-ranked observation rows before the action rows, since the policy reacts to what it sees.
