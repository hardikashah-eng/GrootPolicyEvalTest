# GrootPolicyEvalTest: Robustness, Fault-Tolerance & Data-Fidelity Extensions

*2026-11-07*

> Recovered from the original design document (dated 2026-11-07). Section 6 describes the implementation as of that date; the code now lives in `src/groot_eval` (`degradation/`, `fidelity.py`, `envs/`). All results in it come from the synthetic Arm3 demo with placeholder wear profiles, not from a real robot.

## Summary

All four asks fit the harness without changing its core contract. Each one becomes either an **env wrapper** (same pattern as `RealTimePolicy` in `latency.py`, but on the env side) or a **new scenario parameter family**. Because of that, `sweep`, `compare`, `regress`, `analysis.attribute_failures`, Wilson intervals and `--trace` all work on the new dimensions with no changes. The one real gap is that `Reach2D` is a 2D point mass, so it has no joints that can wear out or lose a DOF. A small articulated CPU env comes first (Phase 0).

| Ask | New module | Hooks into existing | New CLI |
|---|---|---|---|
| Wear and tear by age (new / 1 week / 1 year) and DOF loss | `degradation/wear.py`, `WearEnv` wrapper | `Scenario.params` (`wear.*`), `randomization.py`, `analysis.py` | `groot-eval aging` |
| Subsystem malfunction and multi-fault long-horizon runs | `degradation/faults.py`, `FaultEnv` wrapper | `closed_loop._run_episode` (fault log in `info`), `trace.py` columns | `groot-eval faults` |
| Skill-focused condition suites (walk, grasp, reach) with min / avg / max | `suites/` + `randomization.py` designs (`corners`, `lhs`, `oat`) | `SweepConfig`, named scenarios, `report.py` | `groot-eval suite` |
| Sim-to-real and real-to-sim data fidelity gates | `fidelity/` (gate runner + thresholds YAML) | `checks.py`, `gap.py`, `replay.py`, `correlation.py`, `trace.import-real` | `groot-eval fidelity` |

The design rule throughout: every new effect is **seeded, logged per step, and expressed as scenario parameters**. That way any failure is exactly replayable through `regress`, and attribution can name the cause. A wear effect that you cannot replay or attribute is not useful as a test.

## Phase 0: an articulated CPU env

Add `envs/arm3.py`, a planar arm with 3 to 7 revolute joints, a parallel gripper and a simple pick target. It needs about 250 lines of NumPy and no GPU. Each joint has its own motor model (gain, torque limit, viscous and Coulomb friction, backlash), its own encoder, and a joint-limit range. The point is not realism. The point is that every wear and fault effect below has a **named joint and actuator to act on**, and that CI can exercise them in seconds, the same way `Reach2D` does for latency today.

It should emit observations in GR00T style: `state.joint_pos`, `state.joint_vel`, `state.joint_torque`, `state.gripper`, `state.ee_pose`, and optionally `video.wrist` as a tiny rendered image. That way `checks.py`, `gap.py` and the GR00T adapter run on it unchanged. It should also implement `set_state` so `replay` and sysid work.

Two hooks on every env make the wrappers portable, including to Isaac Lab:

- `components() -> dict[str, ComponentSpec]` lists what can degrade: name, kind (`joint`, `motor`, `encoder`, `camera`, `imu`, `ft_sensor`, `gripper`, `comm`), and which action and state slices it owns.
- `set_component_param(name, key, value)` is an optional physics lever, for effects that cannot be faked at the action or observation boundary (for example, raising joint damping inside the simulator). On Isaac Lab this maps to `param_setters` that write `ArticulationCfg` actuator values (`stiffness`, `damping`, `effort_limit`, `friction`) and joint limits.

For GR00T humanoids, `components()` is generated from the embodiment's `modality.json`: each `state.*` and `action.*` group (left_arm, right_hand, waist, legs) becomes a component group. Then `wear.joint[left_arm.3]` targets the same joint in sim and on the robot.

## 1. Wear and tear model

Model age as a **population, not a point**. A "1-year-old robot" is a distribution of units, because wear depends on duty cycle and load. So each age profile samples per-unit, per-joint wear values from a seeded distribution. A sweep then reports success with an interval for each age, and you can see which joint's wear drives the drop.

### Effects by component

Each effect is applied by `WearEnv`, a wrapper that sits between policy and env (`WearEnv(Arm3Env())`). Commands are degraded on the way in and observations on the way out. Effects that need physics go through `set_component_param`.

| Component | Effect | Where applied | Parameter (per unit, per joint) |
|---|---|---|---|
| Motor | Torque / gain loss (magnet aging, winding heat) | action in | `gain_scale` 1.0 → lower |
| Motor | Lower torque or velocity limit (thermal derate) | action in, clip | `torque_limit_scale`, `vel_limit_scale` |
| Motor | Thermal derate during long episodes | action in, stateful | `thermal_tau_s`, `derate_at_temp` |
| Gearbox / joint | Backlash (dead zone on reversal) | physics or action in | `backlash_rad` |
| Gearbox / joint | Higher Coulomb and viscous friction | physics | `friction_scale`, `damping_scale` |
| Gearbox / joint | Stiction (breakaway torque) | action in | `stiction_nm` |
| Joint | Reduced range of motion | physics (joint limits) | `range_scale` or `[lo, hi]` |
| Joint | **DOF loss**: locked, free-swinging, or partial | physics + action mask | `dof_mode: locked / limp / limited` |
| Encoder | Bias, drift over time, quantization, dropouts | obs out | `bias_rad`, `drift_rad_per_hr`, `quant_bits`, `dropout_p` |
| IMU | Bias random walk, scale error | obs out | `gyro_bias_walk`, `accel_scale` |
| F/T sensor | Offset, gain drift, saturation | obs out | `ft_offset`, `ft_gain`, `ft_sat` |
| Camera | Lens haze, blur, exposure shift, dead pixels, extrinsic shift | obs out (image) | `blur_px`, `exposure_ev`, `dead_px_frac`, `extrinsic_deg` |
| Gripper | Weaker grip, pad wear (lower friction), slower close | action in + physics | `grip_force_scale`, `pad_mu_scale`, `close_time_scale` |
| Comm bus | Added latency and jitter, dropped packets | action in (queue) | reuse `latency.py` models + `drop_p` |

DOF loss needs care. `locked` holds the joint at its current angle and ignores commands. `limp` sets the motor gain to zero so the joint swings under gravity. `limited` shrinks the range. In all three cases the policy still receives commands for that joint, which is how it happens on a real robot. Report whether the policy compensated with the remaining joints; a separate `--inform-policy` flag tests the case where the fault is announced in the observation (`state.health.*`).

### Age profiles

Profiles live in `configs/wear/*.yaml`. Each effect is a function of age in operating hours. The default shape is a bathtub: a small wear-in change, a slow linear phase, then accelerating wear-out. Unit-to-unit spread comes from a log-normal factor.

```yaml
# configs/wear/arm_profiles.yaml
duty_hours_per_day: 8
profiles:
  new:        {age_hours: 0}
  week_1:     {age_hours: 56}
  month_6:    {age_hours: 1460}
  year_1:     {age_hours: 2920}
  end_of_life:{age_hours: 10000}
effects:
  motor.gain_scale:     {curve: linear, at_eol: 0.85, spread: 0.03}
  joint.backlash_rad:   {curve: bathtub, start: 0.0005, at_eol: 0.006, spread: 0.4}
  joint.friction_scale: {curve: bathtub, start: 1.0, at_eol: 1.8, spread: 0.2}
  encoder.drift_rad_per_hr: {curve: linear, at_eol: 0.002}
  camera.blur_px:       {curve: linear, at_eol: 1.5}
  gripper.pad_mu_scale: {curve: linear, at_eol: 0.6, spread: 0.1}
# optional: per-joint load factor (shoulders and hips wear faster)
load_factor: {joint[0]: 1.6, joint[1]: 1.3, default: 1.0}
```

The numbers above are **placeholders**. The real values should come from field data: fleet logs, durability rig runs, and vendor actuator datasheets. A useful side effect is that `replay` sysid (already in the repo) can be pointed at an old robot's trace to fit `gain_scale`, `friction_scale` and `backlash_rad` directly. That is how you calibrate a profile instead of guessing it.

### Same scenario, different ages

The age profile is added as a **paired factor**. The same seed and task parameters run under every profile, so the comparison is within-scenario, not across random draws. That makes small degradations detectable with far fewer episodes.

```bash
groot-eval aging --config configs/reach_pick.yaml \
  --wear configs/wear/arm_profiles.yaml --profiles new,week_1,year_1,end_of_life \
  --env arm3 --policy groot://localhost:5555 --out runs/aging
```

The report contains:

- **Degradation curve:** success rate with Wilson interval per profile, plus task metrics (time to complete, path length, peak torque, tracking error).
- **Paired test:** a McNemar test on the same scenarios across two ages (new vs year_1), which is more sensitive than the two-proportion test `compare` uses today.
- **Robustness margin:** the age at which the lower bound of success crosses the gate (for example 0.8). This is the single number to track across checkpoints.
- **Attribution:** the existing `attribute_failures`, run over the sampled per-joint wear values, to show *which* joint's backlash or which encoder's drift is to blame.
- **DOF-loss matrix:** one row per joint, columns `locked / limp / limited`, cell = success with interval. Joints whose loss drops success to near zero are single points of failure for that skill.

## 2. Fault injection and graceful degradation

Wear changes the robot slowly. Faults change it **mid-episode**. `FaultEnv` wraps any env and applies a seeded **fault schedule**: a list of timed events, each targeting one component. Long-horizon runs draw several faults at random times. The test is not only whether the task succeeds. It is also whether the robot **detects, contains and recovers**, or stops safely.

### Fault catalogue

| Fault | Component | Behaviour once active |
|---|---|---|
| `stuck` | encoder / sensor | reading frozen at last value |
| `dropout` | any sensor | NaN or missing key for N steps |
| `bias_step` | encoder, IMU, F/T | sudden offset |
| `noise_burst` | any sensor | noise x10 for N steps |
| `blackout` | camera | black, frozen or corrupted frames |
| `motor_off` | motor | zero torque (joint goes limp) |
| `motor_runaway` | motor | command saturates in one direction (watchdog test) |
| `joint_lock` | joint | seized at current angle |
| `gripper_fail` | gripper | will not open, or will not close |
| `comm_stall` | bus | actions not delivered for N ms, last command held |
| `latency_spike` | policy link | inference latency jumps (reuses `latency.py`) |
| `power_sag` | all motors | torque limit drops for N steps |

Each fault has `start` (step or seconds), `duration` (or `permanent`), `severity` and `target`. A schedule is either listed explicitly or sampled:

```yaml
# configs/faults/long_horizon.yaml
episodes: 200
max_steps: 3000            # long scenario
faults:
  mode: random
  rate_per_min: 0.5        # Poisson arrivals
  max_concurrent: 2
  catalogue:
    - {type: stuck,       targets: [encoder.*],  duration: [10, 200]}
    - {type: blackout,    targets: [camera.*],   duration: [5, 60]}
    - {type: motor_off,   targets: [motor.*],    duration: permanent, weight: 0.3}
    - {type: comm_stall,  targets: [bus],        duration_ms: [50, 400]}
    - {type: latency_spike, targets: [policy],   latency_ms: [200, 800], duration: [20, 100]}
```

The fault schedule is stored in `Scenario.params["faults"]`, so it is part of `scenario_id`, exported with `--export-suite`, and replayed exactly by `regress`.

### What graceful means: metrics

The runner logs fault onset and end into the trace (`fault.active`, `fault.type`, `fault.target` columns). The report computes:

| Metric | Definition |
|---|---|
| Task success under fault | success rate with interval, split by fault type and by number of concurrent faults |
| Safe outcome rate | success **or** a controlled stop (velocity below threshold, no collision, no joint-limit hit) |
| Unsafe event rate | collisions, joint-limit hits, torque saturation > N steps, self-collision, fall (humanoid) after fault onset |
| Time to react | steps from onset until the action distribution changes significantly (detects that the policy noticed) |
| Recovery time | steps from fault end until tracking error returns to its pre-fault band |
| Blast radius | how much other joints' tracking degrades while one joint is faulted |
| Degradation slope | success vs number of concurrent faults (1, 2, 3); a cliff signals no redundancy |

Gate example: `safe_outcome_rate >= 0.99` (lower bound) for single faults, and `unsafe_event_rate <= 0.01` for any fault. Task success under fault is reported, not gated, because no policy should be required to finish a pick with a seized shoulder.

A policy-side **health monitor** is optional. If the system has one (a watchdog, a state estimator that flags inconsistent sensors), `FaultEnv` can expose `state.health.*` so you can measure detection latency separately from policy reaction.

## 3. Skill-focused condition suites

A suite is one skill (walk, grasp, reach, hand-over, carry) plus the **conditions that stress that skill**, each with explicit `min`, `nominal` and `max` levels. Random sweeps (what `randomization.py` does today) estimate average success well but rarely hit the corners, and the corners are where field failures happen. So suites add three structured designs to `SweepConfig`:

- `oat` (one at a time): every condition at min and max while the others stay nominal. Cheap; it answers "which single condition hurts".
- `corners`: all 2^k combinations of min and max, plus the nominal centre. It finds interactions (fast **and** curved **and** dim). Use for k up to about 6.
- `lhs` (Latin hypercube): space-filling random samples inside the box, for the average and for attribution over continuous ranges.

```yaml
# configs/suites/walk.yaml
skill: walk
task: "walk to the marker"
design: [oat, corners, lhs]
lhs_samples: 200
repeats: 5
conditions:
  speed_mps:          {min: 0.2, nominal: 0.6, max: 1.2}
  surface_curvature:  {min: 0.0, nominal: 0.05, max: 0.3}   # 1/m
  slope_deg:          {min: -10, nominal: 0, max: 10}
  ground_friction:    {min: 0.3, nominal: 0.8, max: 1.2}
  step_height_m:      {min: 0.0, nominal: 0.0, max: 0.08}
  lighting_lux:       {min: 50, nominal: 500, max: 5000}
  payload_kg:         {min: 0, nominal: 0, max: 5}
metrics: [success, time_to_goal, cot, foot_slip, torso_tilt_max, falls]
```

The same schema covers the other skills by changing the condition list:

| Skill | Typical conditions (min / nominal / max) | Skill-specific metrics |
|---|---|---|
| Walk | speed, curvature, slope, friction, step height, payload, lighting | falls, foot slip, cost of transport, torso tilt |
| Grasp | object size, mass, friction, shape class, pose noise, clutter count, lighting, approach speed | grasp success, slip after lift, re-grasps, contact force peak |
| Reach / move hand | target distance, speed, workspace region (edge vs centre), obstacle density | end-effector error, overshoot, jerk, time to reach |
| Carry / hand-over | payload, CoM offset, walking speed, partner timing | drop rate, wrist torque peak |

The report has one table per suite: rows = conditions, columns = success with interval at min, nominal and max, with significant gaps flagged by the existing attribution logic. A **corner heat map** shows the worst 2-way combinations. A suite can also cross with wear and faults (`--wear year_1`), so "walk at max speed on a 1-year-old robot" is a single command.

**Focused datasets.** `--trace` already writes LeRobot v2. With a suite, each episode is tagged with skill and condition levels in `meta/episodes.jsonl`. You get per-skill, per-condition datasets for open-loop scoring, fine-tuning, or to compare against real recordings under the same conditions. `groot-eval suite --coverage REAL_DATASET` also reports which condition cells the real data covers and which only exist in sim.

## 4. Data fidelity: validations and gates

Data should cross the sim/real boundary only through an **ordered set of gates**, cheapest first. A failing gate stops the pipeline, so nobody spends a week tuning physics to fix a degrees-vs-radians bug. Many of the checks already exist in `checks.py`, `trace.import_real`, `gap.py`, `replay.py` and `correlation.py`. The new `fidelity/` package wraps them with thresholds, adds the missing checks, and gives one exit code and one report.

### Gates, in order

| Gate | Question it answers | Checks | Default pass criteria | Exists today? |
|---|---|---|---|---|
| G0 Integrity | Is the file intact and traceable? | schema / `info.json` features, required columns, row counts per episode, no NaN/Inf, monotonic timestamps, SHA-256 per episode, provenance (robot ID, firmware, calibration ID, sim version, git SHA) | zero errors | partial (`import-real`) |
| G1 Contract | Do both sides speak the same language? | keys, dims, units (deg vs rad), scale and offset, gripper range, action-group order, image range and resolution, control rate | zero `error` findings | yes (`check`) |
| G2 Timing | Are signals aligned in time? | control-period jitter, dropped frames, camera-to-state sync offset (cross-correlation), inference latency distribution | jitter p99 < 0.25 dt; sync offset < 0.5 dt; drops < 0.5% | partial (`gap` timing) |
| G3 Physical plausibility | Could a real robot have produced this? | finite-difference velocity vs reported velocity, joint limits and velocity and torque limits respected, commanded vs applied action, ee pose vs forward kinematics of joint pos, contact forces bounded, no teleports | consistency nRMSE < 0.05; zero limit violations | **new** |
| G4 Distribution | Is the sim data from the same world as real data? | per-signal W1 / pooled std, KS, image stats (brightness, contrast, sharpness), embedding-space distance (FID-like) on camera frames, **coverage**: share of real samples inside the sim randomization envelope | max W1 < 0.3 std on state; coverage > 95% | partial (`gap`) |
| G5 Dynamics | Does the sim move like the robot? | replay real actions open loop (`replay`), nRMSE per signal at 1, 10, 50 steps; sysid best-fit not at grid edge; parameters identifiable | nRMSE@10 < 0.1; fit not at edge | yes (`replay`) |
| G6 Outcome | Does sim rank checkpoints as the robot does? | paired outcome agreement, Spearman over checkpoints, per-scenario disagreements | agreement > 85%; Spearman > 0.8 | yes (`correlate`) |

Thresholds live in `configs/fidelity/gates.yaml`, per embodiment. The defaults above are starting points to tune on your own data, not validated limits.

### Direction-specific checks

**Real → sim** (robot logs used to build or calibrate sim, digital twins, or as replay seeds):

- Calibration ID in the log matches the calibration used to convert it; joint zero offsets applied once, not twice.
- Robot-frame vs world-frame conventions (quaternion order wxyz vs xyzw, handedness) checked against a known pose.
- Scene reconstruction: object poses and sizes in the reconstructed scene within tolerance of measured ground truth; mesh watertight; mass and inertia plausible.
- Labels: task strings and success flags agree with a rule-based success check on final state (catches mislabelled episodes).

**Sim → real** (sim data used for training, or sim policies deployed):

- Sim data sits inside the robot's limits (G3 run against the *real* robot's limits, not the sim's).
- Rendering: camera intrinsics, resolution, crop and colour pipeline match the real camera; image-stat and embedding distance within G4 thresholds.
- Action post-processing is identical: the same clipping, filtering and chunk execution mode (`hold` vs `continue`) in sim and on the robot.
- A held-out real validation set: a policy trained on the new sim data must not regress on it (`compare` exit code).

### Using the gates

```bash
groot-eval fidelity --sim traces/sim --real traces/real \
  --gates configs/fidelity/gates.yaml --direction sim2real --out runs/fidelity
# exit 0 = all gates pass, 1 = a gate failed, 2 = a gate could not run (missing data)
```

`fidelity.md` lists each gate with pass, fail or skip, the failing checks, and a likely-cause hint (reusing the hints in `gap.py`). `fidelity.json` is the machine-readable version for CI dashboards. Run G0 to G3 on every new dataset at ingest. Run G4 to G6 when a sim config or a robot changes, and before any sim-trained checkpoint ships.

## 5. CLI, CI wiring and rollout

New commands follow the existing exit-code convention (0 pass, 1 gate failed), so they drop into `.github/workflows/tests.yml` and any checkpoint-promotion pipeline unchanged.

| Command | Gate it can enforce |
|---|---|
| `groot-eval aging` | `--min-success-at year_1 0.8`: lower bound of success at an age profile |
| `groot-eval faults` | `--min-safe 0.99 --max-unsafe 0.01` |
| `groot-eval suite` | `--min-success-corner 0.6`: no min/max corner below this |
| `groot-eval fidelity` | all gates in `gates.yaml` |

Suggested promotion pipeline for a new checkpoint: fidelity G0 to G3 on any new data → existing `sweep` + `compare` + `regress` → `suite` (OAT on every PR, corners nightly) → `aging` (new vs year_1, nightly) → `faults` (single faults on every release, long-horizon multi-fault weekly).

### Phases

1. **Phase 0:** `Arm3` env with `components()` and `set_component_param`.
2. **Phase 1:** `degradation/wear.py` + `WearEnv` + age profiles + `aging` command with paired McNemar test and DOF-loss matrix.
3. **Phase 2:** `degradation/faults.py` + `FaultEnv` + fault metrics + `faults` command.
4. **Phase 3:** `oat` / `corners` / `lhs` designs, suite configs for walk, grasp, reach, and the `suite` command.
5. **Phase 4:** `fidelity/` gate runner, new G0 and G3 checks, coverage metric, and the `fidelity` command.
6. **Phase 5 (GPU):** Isaac Lab `param_setters` for actuator wear and joint limits; humanoid `components()` from `modality.json`; calibrate wear profiles from fleet logs with `replay` sysid.

Phases 0 to 4 run on CPU and can be tested in CI like the rest of the repo. Phase 5 needs an Isaac Lab machine.

### Open questions

- Which robots and skills come first (arm pick, humanoid walk)? That decides the first `components()` mapping.
- Is there fleet or durability-rig data to calibrate age profiles, or do the placeholders stand for now?
- Does the deployed stack have a health monitor whose signals should appear as `state.health.*`?

## 6. Implementation status

Phases 0 to 4 are built on branch `feature/robustness-wear-faults-fidelity` (one commit, 40 files, about 4,000 lines). All 102 tests pass: the 83 existing ones and 19 new ones. `examples/run_robustness_demo.py` reproduces every sample report in about 2 minutes on a CPU. Phase 5 (Isaac Lab levers, humanoid `components()`, fleet calibration) is not started.

Where the build differs from the plan above:

- The fidelity gates are one module, `fidelity.py`, not a package. G0 and G1 are hard prerequisites; G2 to G6 all run so you get the full diagnosis.
- Encoder drift is modelled as a bias that grows with age (`encoder.bias_rad`, random sign per unit), not as drift per hour within an episode.
- Safety gates are strict: `--min-safe` needs the CI *lower* bound to reach the bar, so too few episodes fails. Suite gates fail a cell only when it is *significantly* below the bar.
- Torque saturation is reported but not counted as unsafe. Unsafe means collision, instability or a joint-limit hit after the fault.
- `suite --coverage REAL` (per-cell coverage of real data) is not built. G4 measures coverage in state space instead.

Results on the synthetic Arm3 demo, with placeholder wear profiles:

| Tool | What it found |
|---|---|
| `aging` | 100% through year 1, 97.5% at year 2, 87.0% at year 3 (McNemar p = 3e-8), 78.5% at end of life; margin at 90% falls between year 2 and year 3; joint 1 friction is the only significant driver |
| DOF-loss matrix | limp shoulder 0%, locked shoulder 47.5%, range-limited shoulder 77.5% |
| `faults` (single) | safe outcome 77.8% naive vs 86.5% robust controller; naive sends NaN commands 27 times; `motor_off` is the dominant risk |
| `faults` (long horizon) | success 100% / 84% / 54% with 1 / 2 / 3 concurrent faults |
| `suite` | nothing significant one at a time; `friction=min` with `action_delay=max` drops to 40% at the corners |
| `fidelity` | deg/s velocity stops at G1; worn unit fails G3 (encoder bias past limits), G4 (coverage 92%), G5 (replay error 1.30, or 0.49 after fitting friction, which pins at the edge of its range) |
