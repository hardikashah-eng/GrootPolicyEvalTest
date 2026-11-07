# Data-fidelity gates: FAIL

Direction `real2sim`: sim `demo-output/robustness/traces/sim`, real `demo-output/robustness/traces/robot_fixed`. Gates run in order; after a failure the later gates are skipped, because they would measure the bug instead of the data.

| Gate | Status | Required | Key numbers |
|---|---|---|---|
| G0_integrity | pass | yes | 0 errors, 0 warnings |
| G1_contract | pass | yes | 0 errors, 0 warnings |
| G2_timing | pass | yes | sim: jitter p99 0.0% of a step; real: jitter p99 0.0% of a step |
| G3_plausibility | **FAIL** | yes | 2 errors, 0 warnings |
| G4_distribution | **FAIL** | yes | state gap 0.95 σ; coverage 92.3% |
| G5_dynamics | **FAIL** | yes | nRMSE@10 0.486 |
| G6_outcome | skipped | no | 0 errors, 1 warnings |

## G0_integrity: pass

- ok `sim.manifest`: 64 files match the SHA-256 manifest
- ok `sim.integrity`: 60 episodes, 4510 steps, all finite
- ok `real.manifest`: 64 files match the SHA-256 manifest
- ok `real.integrity`: 60 episodes, 7914 steps, all finite

## G1_contract: pass

- ok `state.keys`: same 6 state keys
- ok `fps`: both at 50 Hz
- ok `summary`: no contract mismatches between sim and real

## G2_timing: pass

- ok `sim.jitter`: p99 deviation from the 20.0 ms period is 0.0% of a step (limit 25%)
- ok `sim.drops`: 0.00% of steps are more than 1.5 periods apart (limit 0.50%)
- ok `real.jitter`: p99 deviation from the 20.0 ms period is 0.0% of a step (limit 25%)
- ok `real.drops`: 0.00% of steps are more than 1.5 periods apart (limit 0.50%)

## G3_plausibility: **FAIL**

- **ERROR** `real.velocity`: d(joint_pos)/dt vs joint_vel: nRMSE 0.502 (limit 0.5); a mismatch means wrong units, a stale velocity filter, timestamps that are not the sample times, or noisy positions
- **ERROR** `real.limits`: joint_pos: 0.71% of steps outside the limits [[-0.2, 3.3416], [-2.6, 2.6], [-2.6, 2.6]]
- ok `sim.velocity`: d(joint_pos)/dt vs joint_vel: nRMSE 0.000 (limit 0.5); a mismatch means wrong units, a stale velocity filter, timestamps that are not the sample times, or noisy positions
- ok `sim.limits`: joint_pos: 0.00% of steps outside the limits [[-0.2, 3.3416], [-2.6, 2.6], [-2.6, 2.6]]
- ok `sim.action_range`: 0.00% of actions outside [-1, 1] (commanded vs applied clipping differs?)
- ok `sim.teleport`: 0 step(s) where a state jumps far beyond its usual step size (0.000%)
- ok `sim.derived`: recorded derived signals vs the env's own computation from the recorded joints: max error 0 (limit 0.01)
- ok `real.action_range`: 0.00% of actions outside [-1, 1] (commanded vs applied clipping differs?)
- ok `real.teleport`: 0 step(s) where a state jumps far beyond its usual step size (0.000%)
- ok `real.derived`: recorded derived signals vs the env's own computation from the recorded joints: max error 0 (limit 0.01)
- ok `real.labels`: 0 episode(s) whose success label disagrees with their final reward (a success that ends worse than the typical failure, or the reverse)

## G4_distribution: **FAIL**

- **ERROR** `state.distance`: largest state gap 0.954 pooled std (gripper; limit 0.3); over the limit: ['gripper', 'joint_torque[0]', 'joint_torque[2]', 'target[1]', 'joint_vel[0]', 'joint_vel[2]']
- **ERROR** `action.distance`: largest action gap 0.79 pooled std (action[3]; limit 0.5); over the limit: ['action[3]', 'action[0]']
- **ERROR** `coverage`: 92.3% of real states fall inside the sim's envelope (limit 95%); most often outside on dims [7, 3, 8]
- ok `coverage.reverse`: 92.4% of sim states fall inside the real envelope (low = sim explores states the robot never visits; informational)

## G5_dynamics: **FAIL**

- warn `sysid`: `friction` = 8 is at the edge of the searched range [1, 8]; widen it
- warn `sysid`: 2 settings score within 5% of the best; not pinned down by this data: `friction`
- **ERROR** `replay`: real actions replayed in sim (best fit {'friction': 8.0, 'payload_kg': 0.2}): normalized state error 0.486 after 10 steps (limit 0.1)

## G6_outcome: skipped

- warn `skipped`: no --pairs CSV of paired sim/real outcomes
