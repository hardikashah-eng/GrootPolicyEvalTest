# Sim-to-real debugging demo (SYNTHETIC)

> **SYNTHETIC.** The "robot" is the Reach2D simulator with hidden physics, sensor noise 0.02, inference latency `normal:110,25` ms executed asynchronously, and a logger that exported velocity x 57.3. It shows the tools finding a gap whose causes are known. It is not a real robot result.

## 1. Contract check

The first export from the robot fails `groot-eval check` with a units error on `velocity` (`check_buggy.md`). After the fix, the check is clean (`check.md`). Comparing physics before this would have compared a unit bug.

## 2. Gap, naive sim vs robot

Largest gap: **0.93 σ** on `state:velocity[1]` (`gap_naive/gap.md`). The timing section reports the robot's 2-3 steps of inference latency that the sim never sees.

## 3. Replay and parameter fit

Replaying the robot's actions in the default sim: tracking error 0.923 nRMSE. The best of 144 settings: 0.038 (`replay/replay.md`).

| Parameter | Hidden (true) | Recovered |
|---|---|---|
| `friction` | 1.0 | 1.0 |
| `actuator_gain` | 0.9 | 0.9 |
| `action_delay` | 1 | 1 |

Replay recovers the actuator's own delay, not the inference latency: recorded actions are the ones the robot applied, so latency is already inside them. Latency is modelled separately in the next step.

## 4. Calibrated sim vs robot

Fitted physics, the known sensor noise, and `--latency-mode trace:traces/real --during-inference continue`.

Largest gap: **0.12 σ** on `state:target[1]` (was 0.93 σ; `gap_calibrated/gap.md`).

## 5. Outcomes per checkpoint

| Checkpoint | Robot | Naive sim | Calibrated sim |
|---|---|---|---|
| ckpt_a | 50% (8/16) | 50% (24/48) | 54% (26/48) |
| ckpt_b | 50% (8/16) | 100% (48/48) | 54% (26/48) |
| ckpt_c | 100% (16/16) | 98% (47/48) | 88% (42/48) |

Outcome agreement on 48 matched checkpoint/scenario pairs: 83% naive, **98% calibrated** (`correlation_before.md`, `correlation_after.md`). Checkpoint ranking agreement (Spearman): 0.00 naive, **1.00 calibrated**.

The naive sim says `ckpt_b` (obstacle avoidance) never fails; on the robot it collides in half the scenarios, because it steers on observations that are 2-3 steps old. Only the sim with the robot's latency shows that.
