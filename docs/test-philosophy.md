# Test philosophy

How grooteval decides whether a GR00T policy, a simulator or a digital twin can be trusted, and why the tools
are built the way they are. The [test plan](gr00t-test-plan.md) lists the test cases and thresholds. This page
covers the reasoning behind them, and maps each idea to the command and the doc that implement it.

## The one question

**Will this policy, on this robot, do the right thing, and how sure are we?**

Everything here exists to answer it with evidence, not with a demo. Two facts shape that evidence:

1. Robot time is scarce and risky, so most testing happens in simulation.
2. Simulation is only useful to the degree that it is *measured* to predict the robot. A sim result is a
   claim about the robot that has to be checked.

The harness therefore has two jobs that never separate. It **tests the policy**, and it **tests the tests**:
the sim, the twin, the data pipeline and the harness itself.

## Ten principles

### 1. Same seed, same run, before anything else

A comparison between sim and real is meaningless if two sim runs of the same seed already disagree. So:

- One **run seed** drives every random source through named, independent streams: scenarios, wear units,
  faults, latency, sensor noise, the scheduler, treadmill perturbations and rig stream faults.
  ([seeding](../src/groot_eval/seeding.py))
- The seed is **written into every trace**, so a replay, a repro or a calibration of a recording draws
  exactly what the original run drew.
- `groot-robot seed-check` runs a seed twice and points to the first step where the runs differ. A
  non-deterministic harness is fixed first, because any noise it adds would otherwise be mistaken for a sim/real
  gap.

### 2. Numbers come with intervals, comparisons are paired

- A success rate is never a bare percentage. It comes with a Wilson interval.
- Checkpoints, timing models, stream conditions and twins are compared **on the same scenarios**, with a paired
  test (McNemar). Two runs on different random scenarios can differ by luck. The same scenarios run both
  ways cannot.
- Every comparison command reports what changed, both ways:
  - `compare` and `regress`: checkpoint A against checkpoint B;
  - `timing-check`: lockstep timing against the robot's timing;
  - `rig`: ideal streams against the rig's streams;
  - `treadmill`: each perturbation type against the others.

### 3. Trust in the sim is measured, gate by gate

Before sim and real data are compared, the data has to cross the boundary intact, so
([transfer verification](transfer-verification.md)) checks these gates in order:

| Gate | Checks |
|---|---|
| G0 integrity | the data is complete and well formed |
| G1 contract | keys, units and scales match |
| G2 timing | the control period, dropped steps and camera sync are the same |
| G3 plausibility | velocities agree with positions, and values stay within limits |
| G4 distribution | each signal is distributed the same way in sim and real |
| G5 dynamics | recorded real actions replayed in sim track the real states |
| G6 outcome | sim and real agree on outcomes and on checkpoint ranking |

Sim-only sign-off is allowed only once outcome agreement and rank correlation pass. Until then, sim results
inform decisions but don't make them.

### 4. Every real failure becomes a twin test

A failure on the robot is the most valuable data there is.

- `groot-twin repro` asks the twin to fail the same way. The verdict is one of: reproduced, reproduced only
  with knob changes, partial, not reproduced, or flaky. ([twin failure repro](twin-failure-repro.md))
- A failure the twin cannot reproduce is a ticket against the twin, with the knob setting that would explain
  it.
- A failure the twin does reproduce goes into the regression suite and is replayed on every later checkpoint.
- The bridge writes watchdog trips straight into repro's record format, so a divergence seen live becomes a
  reproduction case with no manual step.

### 5. The robot is the reference, so profiles come from robots

Placeholder physics produces placeholder answers.

- `groot-robot calibrate` fits each robot's profile from its own traces:
  - system identification;
  - effective age (wear-fit);
  - per-joint wear;
  - timing.

  ([robot profiles](robot-profiles.md))
- Profiles are versioned, hashed, schema-checked and approved. Production code loads only approved versions.
- `fleet-fit` replaces the placeholder wear curves with values fitted across the fleet.
- Each fit records its evidence: the source traces' hashes, and the replay error before and after each stage.
  A stage that does not improve tracking is dropped.

### 6. Model the things that break transfer, one axis at a time

Most sim-to-real failures come from a short list, and each has its own model and its own test:

| What breaks transfer | Model | Test | Doc |
|---|---|---|---|
| Ageing actuators | wear profiles, per joint and per unit; batched on the GPU | `aging`, `actuator-sweep` | [actuators](actuators.md), [robustness](robustness-wear-faults-fidelity.md) |
| Component failures | fault catalogue (encoder, motor, joint, bus, power) | `faults` | [robustness](robustness-wear-faults-fidelity.md) |
| Loop timing | latency, jitter and missed ticks measured from real traces | `timing-profile`, `timing-check` | [scheduler](scheduler.md) |
| Sensor delivery | per-topic rates, delays, drops; stuck and biased streams | `rig` | [rig](rig.md) |
| Ground and disturbances | split-belt treadmill: trips, slips, splits, incline, pushes | `treadmill` | [treadmill](treadmill.md) |
| Physics parameters | system identification over sweep grids | `replay`, `calibrate` | [robot profiles](robot-profiles.md) |
| Data conversion | lossless round trip; HDF5 ↔ LeRobot fidelity | `groot-twin` conversion checks | [conversion](conversion-roundtrip.md) |

Each axis is swept on its own and reported with intervals. A parameter is named as a failure driver only when
its best and worst intervals do not overlap.

### 7. Test the policy as it will actually run

The policy sees stale, jittered, dropped and occasionally wrong inputs. It acts late, through ageing
actuators. Tests that feed it perfect lockstep data overstate it. In the reference examples:

| Condition | Effect |
|---|---|
| Arm controller under the robot's timing (synthetic trace) | success fell from 100% to 1% |
| Walking controller on a treadmill | kept walking after 59% of trips [36, 78], and after every other perturbation type |

The rig records the observation **exactly as the controller received it**, including each topic's age. A
later replay can then reconstruct what the controller saw, not what the plant was doing.

### 8. Shadow before mirror, mirror before deploy

Hardware tests escalate one step at a time ([hardware bridge](hardware-bridge.md)):

1. **Offline shadow**: a recorded robot session is compared with the twin, tick by tick.
2. **Live shadow**: the robot runs its own controller and the twin follows. Nothing is sent to the robot.
3. **Mirror**: the bridge commands the robot (a policy, or replayed captured data), optionally on a subset of
   joints, with a watchdog that stops on divergence, missing or stale samples, limits and command jumps.
4. **Rig on hardware**: fake and real publishers on the real clock, feeding the controller under test (ros2_control
   on mock or real hardware).

The watchdog is a test-rig guard. The robot's e-stop and safety controller are always in charge.

### 9. Fast checks gate every change; expensive checks gate releases

| Tier | When | What runs |
|---|---|---|
| T1 | every PR | unit tests, contract checks, `seed-check`, short CPU sweeps |
| T2 | nightly | full sweeps, regression suite, fidelity gates on new data, treadmill and rig catalogues |
| T3 | release candidate | robustness and safety sweeps, twin paired scenarios, gap report |
| T4 | robot time | shadow, mirror and acceptance, paired with T3 scenarios |

The CPU twins (`arm3`, `walker`, `treadmill`) keep T1 and T2 cheap and deterministic. The same code paths
(array API, batched actuator, scheduler) run in Isaac Lab for T3.

### 10. Be honest about what a test cannot show

Every doc ends with **Limits**, because the most expensive bug is a confident wrong answer. For example:

- the walker is reduced-order;
- the ROS 2 transports have not been run against a robot here;
- gain and friction trade off in replay;
- belt speed does not carry pushes or slips.

A test that passes beyond its limits is reported as untested, not as passed.

## How it fits together

```
           ┌──────────── real robot / treadmill / rig on hardware ────────────┐
           │     traces (seeded, with belt speed, timing, observation ages)   │
           └───────────────┬──────────────────────────────────┬───────────────┘
                 fidelity gates G0–G6                 bridge: shadow / mirror + watchdog
                           │                                  │
                 calibrate → robot profile (approved) ── failures → groot-twin repro
                           │                                  │
             ┌─────────────▼──────────────────────────────────▼──────────────┐
             │  twin: wear · faults · timing · streams · treadmill · sysid   │
             │  sweeps with intervals · paired comparisons · regression suite│
             └─────────────────────────────┬─────────────────────────────────┘
                                           │
                          release verdict for the checkpoint
```

Real data calibrates the twin, and the twin proposes the tests. The robot confirms the highest-value ones, and
every disagreement becomes either a twin ticket or a regression test. The loop tightens over time: each release
needs less robot time for the same confidence.

## What a release decision needs

1. **Determinism:** `seed-check` passes for the configs used.
2. **Data:** the fidelity gates pass on the newest real data.
3. **Twin:** the robot's approved profile is current, and its replay error has not regressed.
4. **Performance:** success meets target on the **lower** confidence bound, and is not significantly worse than
   the previous release on paired scenarios.
5. **Robustness:** survival is reported for each wear age, fault type, timing profile, stream condition and
   perturbation type. A drop of more than 10 points is fixed or documented.
6. **Regressions:** every past failure in the regression suite still passes.
7. **Hardware:** shadow and mirror sessions on the release scenarios show no watchdog trips, or each trip is
   explained and reproduced in the twin.

## Where to start

| You want to | Run | Read |
|---|---|---|
| Know if a checkpoint is better | `groot-eval compare` / `regress` | [test plan](gr00t-test-plan.md) |
| Know if real data can be trusted for sim | `groot-eval fidelity` / `transfer` | [transfer verification](transfer-verification.md) |
| Reproduce a robot failure | `groot-twin repro` | [twin failure repro](twin-failure-repro.md) |
| Make the twin behave like robot sn0042 | `groot-robot calibrate`, `profile approve` | [robot profiles](robot-profiles.md) |
| See if timing breaks the policy | `groot-robot timing-check` | [scheduler](scheduler.md) |
| Stress the policy's inputs | `groot-robot rig` | [rig](rig.md) |
| Test walking under disturbances | `groot-robot treadmill` | [treadmill](treadmill.md) |
| Run the twin next to the robot | `groot-robot bridge` | [hardware bridge](hardware-bridge.md) |
| Check the harness is repeatable | `groot-robot seed-check` | [rig](rig.md#run-seeds) |
