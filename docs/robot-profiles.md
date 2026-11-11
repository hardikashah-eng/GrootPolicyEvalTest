# Robot profiles and the calibration loop

The wear curves in `configs/wear/` are placeholders. This page shows how to
replace them with numbers measured on your own robots, and how to ship those
numbers to production as versioned, checked profiles.

```
robot traces ──► groot-robot calibrate ──► profile (candidate) ──► profile approve ──► production
                                                     │
                     all robots' profiles ──► groot-robot fleet-fit ──► configs/wear/<fitted>.yaml
```

## 1. Calibrate one robot

```bash
groot-robot calibrate traces/sn0042 --robot-id sn0042 --env arm3 \
    --joints shoulder,elbow,wrist --wear configs/wear/arm3_profiles.yaml \
    --sysid configs/arm3_sysid.yaml --operating-hours 2920 --limits configs/robots/arm3.yaml \
    --registry profiles/ --note "after 6-month service"
```

The command runs four stages on the robot's own trace. A stage that does not
lower the replay error is skipped and noted.

| Stage | Uses | Finds |
|---|---|---|
| System identification (`--sysid`) | `replay.system_id` | env physics: friction, actuation delay, gain |
| Effective age (`--wear`) | `wear-fit` (`fit_age`) | the typical unit of the age that best replays the trace |
| Per-joint refinement (`--refine gain,friction`) | coordinate descent on per-joint multipliers | how this unit differs from the fleet median |
| Timing | `TimingProfile.from_trace` | latency, jitter, missed ticks |

On a synthetic test robot (4000 h, with joint 1 given 60% more friction than
its age implies), the replay error fell as follows:

| Twin | Replay error |
|---|---|
| New unit | 1.23 |
| Typical 4545 h unit | 0.28 |
| After per-joint refinement | 0.13 |

Run against the same trace, the bridge in shadow mode gave:

| Twin | nRMSE | Watchdog trips |
|---|---|---|
| New unit | 1.23 | 5 |
| Profiled twin | 0.14 | 0 |

Joint 1's friction came out as the highest of the three joints. Its gain also
came out low, because gain and friction trade off in replay. Treat individual
values as the best explanation of this trace, not as a measurement of each
component. Record varied motions (fast and slow, loaded and unloaded) to
separate them.

## 2. The profile

```yaml
schema_version: 1
robot_id: sn0042
version: 3                 # set by the registry
status: candidate          # candidate | approved | retired (the registry decides)
embodiment: arm3
operating_hours: 2920      # the robot's hour meter, used by fleet-fit
joint_names: [shoulder, elbow, wrist]
control_hz: 50             # limits: same keys as configs/robots/*.yaml
state_limits: {joint_pos: [[-0.2, 3.34], [-2.6, 2.6], [-2.6, 2.6]]}
action_range: [-1.0, 1.0]
max_action_rate: 2.0
actuator:                  # optional
  stiffness: [60, 60, 40]
  motor_model: {path: sn0042_motor.npz, sha256: ...}
wear:
  effective_age_hours: 4545
  values: {gain: [...], friction: [...], backlash: [...], ...}   # one value per joint, joint_names order
  other: {gripper.close_rate_scale: 0.8}                         # wear on other components
sim: {env: arm3, params: {action_delay: 1}}
timing: {control_hz: 50, latency_ms: {...}, ...}                 # TimingProfile format
faults: {rates_per_hour: {motor_off: 0.0001}}                    # optional
provenance:
  traces: [{path: traces/sn0042, sha256: ..., episodes: 40, steps: 9120, domain: real}]
  fit: {baseline_score: 1.23, score: 0.13, stages: [...]}
  notes: [...]
content_hash: sha256:...   # over everything but version and status
```

Each load validates the profile against the schema:

- known keys only;
- one value per joint;
- wear values within plausible ranges;
- known fault kinds;
- a `timing` section that loads as a `TimingProfile`.

Loading also checks the content hash, so a hand-edited file is refused.
`groot-robot profile validate` lists every problem it finds.

## 3. Registry and approval

```bash
groot-robot profile list    --registry profiles/
groot-robot profile diff    registry:profiles/:sn0042@2 registry:profiles/:sn0042@3
groot-robot profile approve sn0042 3 --registry profiles/ --by qa-lead
groot-robot profile get     sn0042 --registry profiles/ --out deploy/sn0042.yaml   # the approved version
```

The registry stores files under `<root>/<robot_id>/v0001.yaml`. An index,
`registry.json`, records each version's hash, its fit score, and the approval
history.

Published files are never rewritten. If a file no longer matches the hash
recorded at publish time, loading it fails.

Production code loads the approved version through
`ProfileRegistry(root).resolve(robot_id)` or the spec
`registry:<root>:<robot_id>@approved`. Calibration only ever adds candidates.

## 4. Using a profile

| Consumer | How it loads the profile |
|---|---|
| Isaac Lab actuator | `WearAwareActuatorCfg(robot_profile="deploy/sn0042.yaml")`. Wear is read by joint name with `wear_for(joint_names)`, so the simulator's joint order may differ from the robot's. |
| Scheduler | `TimingProfile.load("deploy/sn0042.yaml")` (it reads the `timing` key), or `groot-robot timing-check --profile deploy/sn0042.yaml` |
| Hardware bridge | `groot-robot bridge --twin arm3 --twin-profile registry:profiles/:sn0042` |
| CPU twin in code | `env, params = profile.make_twin()`, then `Scenario(seed, params=params)` |
| Transfer checks | `groot-eval transfer --robot deploy/sn0042.yaml` (the profile carries the limit keys) |

## 5. Fleet fit: replace the placeholder curves

```bash
groot-robot fleet-fit --registry profiles/ --which approved \
    --wear configs/wear/arm3_profiles.yaml --out configs/wear/arm3_fleet.yaml
```

Each effect in the base config keeps its curve shape. Its `at_eol` value is
refitted by least squares on the curve, using one point per robot and joint
(operating hours, measured value). Its `spread` is the scatter between units.

`arm3_fleet.md` reports the number of points and the R² for each effect. An
effect with fewer than two robots past zero hours keeps its placeholder.
Robots without `operating_hours` are skipped.

## Limits

- Replay can only see wear that changes how the robot responds to commands:
  gain, torque limit, friction, damping, backlash, stiction and thermal
  derating. Calibration does not fit signed encoder biases or noise; compare
  those with `groot-eval gap`.
- Per-joint refinement adjusts gain, torque limit, friction and damping
  multipliers only. Backlash and stiction stay at the fitted age's values.
- No learned motor model is fitted here. A profile can reference one trained
  elsewhere, by path and sha256.
- The CPU twin is the reference used throughout; the Isaac Lab path consumes
  the same numbers but has not been run here.
