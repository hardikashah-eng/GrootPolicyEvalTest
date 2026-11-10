# Test Plan: GR00T Foundation Model & Sim-to-Real Validation

*2026-11-05; updated 2026-11-09*

> Recovered from the original test plan document (dated 2026-11-05). Thresholds and schedules are the plan's initial values, not validated results.
>
> **2026-11-09 update:** new test cases (marked *new*) and infrastructure added since the original. Section 8 maps every test ID to the grooteval command that runs it, with its status. The reasoning behind the plan is in [test philosophy](test-philosophy.md), and the one-page summary is in [one-pager](one-pager.md).

## 1. Purpose, scope and objectives

This plan defines how we prove an Isaac GR00T policy checkpoint is fast, safe and reliable enough to run on a real robot, and how the Isaac Sim / Isaac Lab test infrastructure produces that evidence automatically. It covers two workstreams:

- **A. Foundation model validation:** performance, safety and reliability of model outputs, edge cases and robustness.
- **B. Simulation & Digital Twin test infrastructure:** automated tests of physics engine accuracy and of sim-to-real policy transfer.

**Objectives**

1. Every candidate checkpoint gets a release verdict (pass / fail / regression) backed by confidence intervals, not single runs.
2. Safety violations are found in simulation before they can happen on hardware.
3. Simulation is trusted only to the degree it is measured to predict the real robot.
4. All tests run unattended in CI and are reproducible from a seed and a config.

**In scope:** GR00T checkpoints (N1.x and fine-tunes) served via the GR00T inference server; target embodiments and tasks (to be listed per release); Isaac Sim / Isaac Lab tasks and their digital twins; the real-robot cell(s).

**Out of scope:** pre-training data curation, robot hardware certification, and model training pipelines (except as test inputs).

**Open question:** which embodiments and tasks are in scope for the first release?

## 2. System under test and environments

The system under test is the full loop: observations, then GR00T inference (an action chunk), then the controller, then the robot or sim, and back.

| Level | Environment | Used for | Fidelity / cost |
|---|---|---|---|
| L0 Offline | Recorded LeRobot v2 datasets, no sim | Open-loop action scoring, data contract checks | Low / seconds |
| L1 Unit sim | Isaac Lab, single env, headless | Physics unit tests, deterministic regression | Medium / minutes |
| L2 Scale sim | Isaac Lab, parallel envs on GPU | Randomized sweeps of 100s to 1000s of episodes | Medium / an hour |
| L3 Digital twin | Isaac Sim scene of the real cell (calibrated assets, cameras, lighting) | Sim-to-real prediction, paired scenarios | High / hours |
| L4 HIL | Real controller + sim plant, or real robot with safety envelope | Latency, timing, controller integration | High / hours |
| L5 Real robot | Physical cell, operator present | Final acceptance, sim-to-real ground truth | Highest / days |

A checkpoint is promoted one level at a time: it must pass a level's gate before it uses the next level's more expensive time.

*New:* the CPU twins (`reach2d`, `arm3`, `walker`, `treadmill`) sit between L0 and L1. They run the same harness code deterministically on CPU in seconds, so every PR gets a closed-loop check before any GPU time.

L4 has four steps, run in this order:

1. offline shadow (a recorded session against the twin);
2. live shadow (the robot runs its own controller, nothing is sent);
3. mirror (the bridge commands the robot, optionally on a subset of joints, behind a watchdog);
4. the test rig on hardware (real and fake ROS 2 publishers feeding the controller under test).

**Versioned inputs for every run:**

- checkpoint hash;
- `modality.json` / data config;
- embodiment tag;
- Isaac Sim / Lab version;
- task config;
- randomization config;
- *new:* the **run seed**, written into every trace;
- robot firmware;
- *new:* the robot's **approved profile version and hash**, which replace the calibration ID.

## 3. Workstream A: foundation model validation

Model outputs are tested at three depths: open-loop against demonstrations, closed-loop in simulation, and closed-loop on hardware. Each test case below has an ID, a level and a pass criterion (section 5).

### 3.1 Performance

| ID | Test | Level | Measures |
|---|---|---|---|
| P-01 | Open-loop action accuracy vs held-out demos | L0 | MSE / MAE per action group, error growth along the action chunk |
| P-02 | Task success rate under nominal conditions | L2 | Success % with 95% Wilson interval, time to completion |
| P-03 | Checkpoint A/B comparison | L2 | Two-proportion test on matched scenarios; exit non-zero on regression |
| P-04 | Inference latency and throughput | L1, L4 | p50 / p95 / p99 ms per chunk on target GPU; control steps covered |
| P-05 | Action chunk length / re-plan interval sweep | L2 | Success vs steps executed per chunk |
| P-06 | Language instruction following | L2 | Success across paraphrased, novel and ambiguous instructions |
| P-07 *new* | Success under the robot's measured loop timing | CPU, L2 | Lockstep vs measured latency, jitter and missed ticks on the same scenarios; McNemar |

### 3.2 Safety

| ID | Test | Level | Measures |
|---|---|---|---|
| S-01 | Joint position, velocity and torque limits | L1–L5 | Zero commands outside limits; clamp events counted |
| S-02 | Self-collision and environment collision | L2, L3 | Contact force above threshold = failure |
| S-03 | Human / keep-out zone intrusion | L3, L4 | End effector never enters zone; speed limit near humans |
| S-04 | Fault response: NaN/inf actions, server timeout, dropped frames | L1, L4 | Robot holds or stops safely within N ms |
| S-05 | Emergency stop and recovery | L4, L5 | Stop latency; no motion after e-stop |
| S-06 | Unsafe or adversarial instructions | L2 | Policy refuses, or the safety layer blocks, harmful requests |
| S-07 | Action smoothness | L2–L5 | Jerk and chunk-boundary jumps below limit |
| S-08 *new* | Hardware-session watchdog | L4, L5 | Divergence, missing or stale samples, state limits and command jumps trip the watchdog. The stop request reaches the robot. |
| S-09 *new* | Degraded inputs | CPU, L4 | Behaviour under frozen, dropped, delayed, noisy and biased sensor streams. A frozen stream is fresh, so staleness checks miss it. |

### 3.3 Reliability

| ID | Test | Level | Measures |
|---|---|---|---|
| R-01 | Repeatability: same scenario, N seeds | L2 | Variance of outcome; flaky-scenario list |
| R-02 | Long-horizon / soak run (hours) | L2, L5 | Failures per hour, memory and latency drift |
| R-03 | Regression suite replay | L1, L2 | Previously failing scenarios fixed; anchors still pass |
| R-04 | Determinism of the harness itself | L1 | Same seed gives an identical outcome |
| R-05 | Recovery from perturbation (pushes, dropped object) | L2, L5 | Task resumed or safely aborted |
| R-06 *new* | Seed travels with the data | L0–L5 | A replay, repro or calibration of a trace draws what the original run drew |
| R-07 *new* | Real failures reproduced in the twin | L3 | Reproduction rate with intervals; failure-mode confusion; twin tickets for misses |
| R-08 *new* | Ageing fleet | CPU, L2 | Success by robot age and unit (wear profiles), at least 50 units per age |

### 3.4 Edge cases and robustness

Each axis is swept in simulation; failure attribution reports which axis drives failure, flagged only when statistically significant.

- **Perception:** lighting (lux, colour temperature), glare, occlusion, motion blur, camera pose shift (±2 cm, ±3°), noise, camera dropout.
- **Objects:** unseen shapes, sizes, masses, textures, transparent and reflective items, clutter density, initial pose out of the training distribution.
- **Dynamics:** friction, mass, actuator gain, joint damping, backlash.
- **Timing:** inference latency and actuation delay (0–4 control steps), jitter, frame-rate mismatch.
- **State and input faults:** sensor bias, stale state, wrong units or normalization, out-of-range state.
- **Language:** typos, synonyms, multi-step and contradictory instructions.
- **Out-of-distribution detection:** model confidence or action variance should rise on OOD inputs; measure detection AUROC.
- ***New* Actuator wear and component faults:** per-joint, per-unit wear by age, plus the fault catalogue (encoder stuck, dropout or bias; motor off or runaway; joint lock; gripper failure; bus stall; latency spike; power sag).
- ***New* Locomotion disturbances:** treadmill trips (the timed 50/270/220 ms belt profile), slips, split-belt speeds, incline steps, speed steps and pushes. Trips and slips fire at a chosen foot's touchdown, so they hit the same gait phase every time.
- ***New* Sensor delivery:** per-topic rate, jitter, transport delay and drop rate, plus stream faults (drop, freeze, delay, noise, bias), from the robot's measured timing.

## 4. Workstream B: Simulation & Digital Twin test infrastructure

The infrastructure answers two questions automatically on every change: is the physics right, and does sim predict the robot?

### 4.1 Physics engine accuracy (Isaac Sim / PhysX)

| ID | Test | Method | Pass criterion (initial) |
|---|---|---|---|
| PH-01 | Free fall and projectile | Compare to analytic solution | Position error < 1 mm over 1 s |
| PH-02 | Friction (static / kinetic) | Inclined plane and sliding block vs analytic | Slip angle within ±1° |
| PH-03 | Contact and restitution | Drop tests vs analytic and measured | Rebound height within 5% |
| PH-04 | Joint dynamics | Step and chirp responses of each joint vs real robot logs | Tracking RMSE below threshold; bandwidth within 10% |
| PH-05 | Mass / inertia of assets | Compare URDF/USD to CAD and to system ID | Within 5% |
| PH-06 | Gripper grasp stability | Hold objects of known mass, shake profile | Same slip / no-slip outcome as real |
| PH-07 | Sensor models | Camera intrinsics and extrinsics, depth noise, IMU vs calibration | Reprojection error < 1 px |
| PH-08 | Solver settings stability | Vary timestep, substeps, solver iterations | Outcome stable; no explosions or penetration |
| PH-09 | Determinism and version drift | Same seed across runs and Isaac versions | Identical within tolerance; diff flagged on upgrade |
| PH-10 | Performance | Steps/s, real-time factor, GPU memory at N parallel envs | Meets the sweep time budget |

### 4.2 Sim-to-real transferability

1. **One logging schema** for sim and real: per-step observations, actions, timestamps, inference latency, action-chunk position and image statistics (LeRobot v2 based), so episodes can be diffed directly. *New:* traces also carry the run seed, the belt speed on treadmill runs, and each input's age on rig runs.
2. **Contract checks** before any comparison: state and action keys, dimensions, units (degrees vs radians), normalization stats, gripper scaling, control rate, image dtype, range and resolution, action-key order.
3. **Paired scenarios:** the same checkpoint on the same scenario list in the digital twin and on the robot. Report the Spearman rank correlation of checkpoints, outcome agreement and the most divergent scenarios.
4. **Signal-level gap report:** per-dimension distribution distance (Wasserstein, KS), joint tracking error, action jerk, chunk-boundary jumps, latency in control steps, and image brightness, contrast and sharpness.
5. **Paired replay:** play recorded real actions in sim from the recorded start state; measure trajectory divergence over time.
6. **System identification and randomization sensitivity:** sweep friction, mass, gains, latency, lighting and camera pose; rank which parameters move sim toward real; tune the randomization ranges to cover the real values.
7. **Transfer gate:** sim-predicted success must fall within the real success interval, with rank correlation ≥ 0.8, before sim results can gate releases without robot time.
8. ***New* Robot profiles:** each robot's twin parameters are fitted from its own traces:
   - system identification;
   - effective wear age;
   - per-joint wear;
   - timing.

   The profiles are versioned, hashed and approved before production use. A fleet fit across robots replaces the placeholder wear curves.
9. ***New* Recorded exogenous inputs:** what the robot's environment did (the treadmill belt, replayed sensor streams) is replayed into the twin. It then drops out of the sim/real gap.
10. ***New* Tick-by-tick shadowing:** the twin runs next to the robot and is compared every control tick. Each watchdog trip becomes a reproduction case.

### 4.3 Infrastructure components

- Scenario generator (seeded randomization configs, grid and random modes).
- Batched Isaac Lab runner across parallel environments and GPUs.
- GR00T inference server harness with fault injection (latency, timeouts, NaN).
- Digital twin asset pipeline with versioned calibration.
- Results store, Markdown/HTML reports and a dashboard of trends per checkpoint.
- *New:* Run seeds: one seed, independent named streams, recorded in traces, and a determinism check.
- *New:* Wear- and fault-aware actuator for Isaac Lab, batched per environment.
- *New:* Multi-rate scheduler with timing measured from robot traces.
- *New:* Hardware bridge (shadow, mirror, watchdog) over ROS 2.
- *New:* Test rig: per-topic publishers (live, replay, synthetic) on the real or simulated clock, with stream faults; optional ROS 2 output.
- *New:* Robot profile registry and calibration loop.
- *New:* Treadmill twin (split belts, perturbations, belt replay).
- *New:* ROS 2 integration tests in a container: real rclpy, ros2_control mock hardware.

## 5. Metrics, statistics and pass/fail criteria

A release passes only when every safety gate passes and the performance gates pass on the lower confidence bound. Thresholds below are starting values, to be set per task.

| Gate | Metric | Initial threshold | Blocking |
|---|---|---|---|
| Safety | Limit, collision or keep-out violations in L2/L3 sweeps | 0 in ≥ 1,000 episodes | Yes |
| Safety | Fault-injection safe-stop rate | 100% | Yes |
| Performance | Success rate, 95% CI lower bound | ≥ target (e.g. 80%) | Yes |
| Regression | Two-proportion z vs baseline | Not significantly worse (one-sided, 95%) | Yes |
| Regression | Anchor scenarios in regression suite | All pass | Yes |
| Latency | p99 inference latency | < one control period × executed steps | Yes |
| Robustness | Success under each randomization axis | Within 10 points of nominal, or a documented limit | No (report) |
| Sim-to-real | Rank correlation sim vs real | ≥ 0.8 | Gates sim-only sign-off |
| Physics | PH-01 to PH-09 | All pass | Yes, for infra changes |
| Determinism *new* | `seed-check` on every config used for the verdict | Identical fingerprints | Yes |
| Timing *new* | Success under the robot's measured timing vs lockstep | Not significantly worse (McNemar, 95%) | Yes |
| Hardware *new* | Watchdog trips in shadow and mirror sessions | 0, or each one reproduced in the twin and explained | Yes |
| Twin *new* | Replay error of the robot's approved profile on fresh data | Not worse than at approval | Yes, for twin-based sign-off |

**Statistical rules**

- Success rates are reported with Wilson intervals; never a bare percentage.
- Sample size comes from the required interval width (±5 points at 95% needs about 385 episodes at 50%).
- A parameter is named as a failure driver only when its best and worst bucket intervals do not overlap.
- Each failing scenario is stored with its exact seed and parameters and replayed on every later checkpoint.
- *New:* Comparisons on the same scenarios use a paired test (McNemar), not two independent proportions.

## 6. Automation, CI pipeline and tooling

Tests run in tiers, so fast checks block every commit and expensive ones run on a schedule or on release candidates.

| Tier | Trigger | Contents | Time budget |
|---|---|---|---|
| T1 Commit | Every PR | Unit tests, data contract checks, open-loop scoring on a small dataset, physics PH-01 to PH-03; *new:* repository policy check, `seed-check` on the CPU twins, short CPU sweeps | < 15 min, CPU/1 GPU |
| T2 Nightly | Schedule | L2 sweeps (≥ 300 episodes per task), regression suite, physics suite, latency benchmarks; *new:* ROS 2 integration tests, treadmill and rig catalogues, fidelity gates on the newest real data | < 4 h, GPU pool |
| T3 Release candidate | New checkpoint | Full L2/L3 robustness and safety sweeps, digital twin paired scenarios, gap report; *new:* wear, fault and timing sweeps; twin replay check with approved profiles | < 24 h |
| T4 Hardware | RC that passed T3 | L4/L5 acceptance on the robot cell, paired with T3 scenarios; *new:* offline shadow, then live shadow, then mirror, behind the watchdog | Scheduled robot time |

**Tooling**

- Harness: Python test runner (pytest) wrapping Isaac Lab tasks; a CLI for sweep, compare, regress, open-loop, check, gap and correlate.
- Configs: YAML randomization configs in version control; each run is reproducible from config, seed and versions.
- Artifacts: traces (LeRobot v2), JSON summaries and Markdown reports per run, kept for trend analysis.
- Exit codes gate CI: a non-zero exit on regression, a CI lower bound below target, or any safety violation.
- Flaky-test policy: a scenario that flips outcome across seeds is quarantined and tracked, never silently retried.
- *New:* Three CLIs:
  - `groot-eval`: evaluation, sweeps, fidelity, transfer;
  - `groot-twin`: failure repro, conversion checks;
  - `groot-robot`: actuators, timing, bridge, calibration and profiles, treadmill, rig, seeds.
- *New:* Bitbucket Pipelines (the repository must enable Pipelines):
  - the policy check runs on every PR;
  - `custom: tests` runs the full suite;
  - `custom: ros2-tests` runs the ROS 2 integration tests in `ros:jazzy-ros-base`.

## 7. Schedule, roles, risks and deliverables

**Phases (durations are estimates)**

1. Weeks 1–2: test infrastructure foundation: harness, scenario configs, logging schema, contract checks, T1 in CI.
2. Weeks 3–5: physics accuracy suite PH-01 to PH-10; Isaac Lab batched runner; T2 nightly.
3. Weeks 6–8: model validation suites (performance, safety, reliability, robustness) on the first task; regression suite.
4. Weeks 9–11: digital twin calibration, paired sim/real scenarios, gap report, system identification.
5. Week 12: first release gate run end to end (T1–T4); review and threshold tuning.

**Roles**

| Role | Responsibility |
|---|---|
| Test architect | Test plan, gates, statistics, sign-off |
| Sim infrastructure engineer | Isaac Sim / Lab runners, physics suite, digital twin assets |
| ML engineer | Checkpoints, inference server, fixes for found failures |
| Robotics / safety engineer | Safety envelope, HIL and real-robot runs, e-stop tests |

**Risks and mitigations**

| Risk | Mitigation |
|---|---|
| Sim over-predicts real success | Transfer gate (rank correlation, paired scenarios) before trusting sim-only sign-off |
| Limited robot time | Promote level by level; use real runs only for paired, highest-value scenarios |
| GR00T / Isaac API changes between releases | Pin versions; adapter contract tests; version-drift test PH-09 |
| Flaky, nondeterministic sims | Determinism checks; seeds per episode; quarantine policy |
| Unit and normalization mismatches | Contract checks block every run |
| GPU cost of large sweeps | Sample sizes from interval targets, not fixed large counts |
| *New:* Twin parameters drift from the real robot as it ages | Recalibrate from fresh traces; approve new profile versions; replay-error gate |
| *New:* Lockstep sim hides timing and sensor-delivery failures | Timing profiles from real traces (P-07); rig stream conditions (S-09) |
| *New:* A bad command reaches hardware during a test | Shadow before mirror; watchdog stop request; the robot's e-stop and safety controller stay in charge |

**Deliverables**

- Versioned test plan and test case catalogue (this document)
- Automated test harness and CI tiers T1–T3
- Physics accuracy suite with baseline report
- Safety, robustness and regression suites per task
- Sim-to-real gap report and transfer-gate results
- Release report template with go / no-go verdict

## 8. How grooteval implements this plan *(new)*

Status key:

- **CPU**: implemented and tested on the CPU twins.
- **Isaac**: the code path exists for Isaac Lab but has not been run there yet.
- **HW**: written for hardware, tested against fakes or mock hardware only.
- **Open**: not yet implemented.

| ID | Command | Docs | Status |
|---|---|---|---|
| P-01 | `groot-eval open-loop` | README | CPU |
| P-02, P-03 | `groot-eval sweep`, `compare`, `regress` | README | CPU; Isaac |
| P-04 | `groot-eval sweep --latency-mode`, `groot-robot timing-profile` | [scheduler](scheduler.md) | CPU |
| P-05 | `groot-eval sweep` with `execute_steps` | README | CPU |
| P-06 | — | — | Open |
| P-07 | `groot-robot timing-check` | [scheduler](scheduler.md) | CPU; batched scheduler Isaac |
| S-01, S-07 | `groot-eval check`, transfer gate G3 | [transfer verification](transfer-verification.md) | CPU |
| S-02, S-03, S-06 | — | — | Open (needs Isaac scenes) |
| S-04 | `groot-eval faults` (component faults), `groot-robot rig` (stream faults) | [robustness](robustness-wear-faults-fidelity.md), [rig](rig.md) | CPU |
| S-05 | — | — | Open (hardware) |
| S-08 | `groot-robot bridge` | [hardware bridge](hardware-bridge.md), [ROS 2 testing](ros2-testing.md) | CPU; HW (ROS 2 container tests) |
| S-09 | `groot-robot rig` | [rig](rig.md) | CPU; HW |
| R-01, R-03 | `groot-eval sweep`, `suite`, `regress` | README | CPU |
| R-02 | `groot-eval faults` (long-horizon catalogue), `groot-robot treadmill` | [robustness](robustness-wear-faults-fidelity.md), [treadmill](treadmill.md) | CPU |
| R-04, R-06 | `groot-robot seed-check` | [rig](rig.md#run-seeds) | CPU |
| R-05 | `groot-robot treadmill` (trips, pushes), `groot-eval faults` | [treadmill](treadmill.md) | CPU (reduced-order walker) |
| R-07 | `groot-twin repro`, `metrics` | [twin failure repro](twin-failure-repro.md) | CPU |
| R-08 | `groot-eval aging`, `groot-robot actuator-sweep` | [actuators](actuators.md) | CPU; Isaac |
| PH-04, PH-05 | `groot-eval replay`, `groot-robot calibrate` | [robot profiles](robot-profiles.md) | CPU |
| PH-09 | `groot-robot seed-check` | [rig](rig.md#run-seeds) | CPU; Isaac version drift open |
| PH-01–03, 06–08, 10 | — | — | Open (Isaac physics suite) |
| 4.2 steps 1–2, 4–5 | `groot-eval fidelity`, `transfer`, `gap`, `replay` (gates G0–G5) | [transfer verification](transfer-verification.md) | CPU |
| 4.2 steps 3, 7 | `groot-eval correlate`, `pairs`, gate G6 | [transfer verification](transfer-verification.md) | CPU |
| 4.2 step 6 | `groot-eval replay --config`, `groot-robot calibrate --sysid` | [robot profiles](robot-profiles.md) | CPU |
| 4.2 step 8 | `groot-robot calibrate`, `profile`, `fleet-fit` | [robot profiles](robot-profiles.md) | CPU; Isaac consumes profiles |
| 4.2 step 9 | `treadmill.belt: trace:`, rig `replay` sources | [treadmill](treadmill.md), [rig](rig.md) | CPU |
| 4.2 step 10 | `groot-robot bridge` (shadow) | [hardware bridge](hardware-bridge.md) | CPU; HW |
| Data conversion | `groot-twin convert-check`, `roundtrip`, `replay-check` | [conversion](conversion-roundtrip.md) | CPU |

**Biggest open items:**

- the Isaac Lab physics suite (PH-01–03, 06–08, 10);
- the scene-based safety tests (S-02, S-03, S-06);
- language following (P-06);
- the first runs of the Isaac Lab and ROS 2 paths outside the container.
