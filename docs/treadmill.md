# Treadmill testing for walking

This page covers the twin of a humanoid on an instrumented treadmill. The
twin lets you rehearse and score walking tests before they run on the real
rig. A recorded belt from the real rig can then be replayed into the twin,
so the belt drops out of the sim/real gap.

## The twin

| Env | What it is |
|---|---|
| `walker` | Reduced-order humanoid walking overground (sagittal plane) |
| `treadmill` | The same walker on a split-belt treadmill |

The walker is a linear inverted pendulum (CoM at constant height) with two
feet on a step clock:

- The stance foot rides its belt.
- The ankle can place the centre of pressure only inside the foot.
- The swing foot moves at a limited speed and lands where it is when the step
  ends.
- Friction caps the ground force, and the foot slips beyond it.
- Incline and external pushes act on the CoM.

The robot falls (`fall`) when it leans too far over its stance foot. It fails
with `off_belt` when it leaves the deck. It succeeds after `duration_s`
(20 s) of walking.

Two reference controllers are included:

| Policy | Behaviour |
|---|---|
| `walker-capture` | Capture-point stepping. Walks at the belt speed minus a correction toward the deck centre. |
| `walker-blind` | The same stepping at a fixed speed, ignoring the belt. It walks off unless the treadmill recenters. |

The twin is reduced-order: there are no joints, no lateral balance and no
foot rotation. Use it to design tests, check the rig logic and replay belts.
It does not predict how a specific humanoid recovers. That needs the full
model in Isaac Lab.

## Treadmill settings

All settings are `treadmill.*` scenario parameters, so sweeps, system
identification, repro and the bridge set them like any other parameter.

| Parameter | Default | Meaning |
|---|---|---|
| `speed` | 0.8 | belt speed, m/s |
| `ramp_s` | 2 | ramp from standstill |
| `mode` | fixed | `fixed`, or `recenter` (belt speed follows the robot's deck position with `kp`, `kd`) |
| `accel_limit` | 3 | belt motor acceleration, m/s² |
| `deck_length` | 1.6 | usable deck, m |
| `split` | 1.0 | right/left belt speed ratio |
| `incline_deg` | 0 | deck slope |
| `perturbations` | [] | scripted events, see below |
| `belt` | — | replay recorded belt speeds: `trace:<dir>` (episode matched by seed) or a list of `[left, right]` |

## Perturbations

| Type | Severity | What happens |
|---|---|---|
| `trip` | belt acceleration, m/s² | One belt slows for 50 ms, surges for 270 ms, then returns over 220 ms. This is the timed trip protocol from instrumented split-belt treadmills. |
| `slip` | friction coefficient | One belt becomes slippery for `duration_s` |
| `speed_step` | m/s | Both belts change speed |
| `split` | ratio | One belt runs faster or slower |
| `incline_step` | degrees | The deck tilts |
| `push` | N | A horizontal push on the CoM (gantry pusher) |

Trips and slips start at the next touchdown of their side after `start_s`,
so they always hit early stance. Use `trigger: time` to start at an exact
time instead.

A catalogue (`configs/treadmill/*.yaml`) samples one schedule per scenario.
Each schedule is fixed by `--seed` and the scenario's own seed:

- Rerunning gives the same trips at the same points in the gait.
- Adding scenarios leaves the existing ones unchanged.

```bash
groot-robot treadmill --config configs/treadmill/single_perturbation.yaml --episodes 60 --record --out runs/tm
```

The command reports, for each perturbation type, the share of episodes the
robot kept walking (with a Wilson interval), the failure modes, the maximum
lean and how far the robot drifted on the deck. On the reference controller
at 0.8 m/s with seed 0, it recovered from every push, slip, speed step,
split and incline step. It recovered from 59% of trips [36, 78]; every trip
failure was a fall.

`trip_ladder.yaml` runs only trips, from mild to severe, to find the
recovery limit. The reference controller recovers at 2–4 m/s² and falls by
14 m/s².

## Recorded belt speeds: sim2real and real2sim

Every treadmill episode records `state.belt_speed` (left and right), so
traces carry the belt.

- **Sim2real.** Run the same seeded perturbation schedule on the real rig.
  The rig's controller executes it, and its trace records what the belt
  actually did.
- **Real2sim.** Replay the robot's commands in the twin with the recorded
  belt, not a nominal one:

```bash
groot-robot treadmill-replay traces/rig_session --out runs/replay
```

The replay resyncs to the recorded state every 10 steps. Walking is
unstable, so an open-loop replay lets any difference grow exponentially.

On a recentering run, replay error was:

| Belt in the twin | Replay error |
|---|---|
| Nominal | 0.60 |
| Recorded | 0.0001 |

On the perturbation sweep, replay error was:

| Belt in the twin | Replay error |
|---|---|
| Nominal | 0.139 |
| Recorded | 0.045 |

What remains on the sweep is the pushes, slips and incline steps, which the
belt speed does not carry.

The bridge takes the same parameter, for offline shadow against a recorded
session:

```bash
groot-robot bridge --robot trace:traces/rig_session --twin "treadmill:treadmill.belt=trace:traces/rig_session" --out runs/shadow
```

## Limits

- Belt replay matches episodes by scenario seed, so the rig's trace has to
  record the seed. Live ROS 2 sessions do not yet stream belt speed into the
  twin tick by tick.
- Slip, push and incline events are not recoverable from belt speed alone.
  Log them as events on the rig if the twin should replay them.
