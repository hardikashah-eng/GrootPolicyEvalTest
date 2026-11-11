# grooteval: one page

**What it is.** A test harness that decides, with evidence, whether a GR00T policy will do the right thing on
a real robot. It also checks the simulator, the digital twin and the data pipeline, because sim results count
only as far as they are measured to predict the robot.

**Why it matters.** Robot time is scarce and risky, so most testing has to happen in sim. Sim overstates
policies unless it carries the robot's real imperfections: worn actuators, failing parts, late and jittery
loops, stale sensor data, uneven ground. grooteval models each of these and measures how far to trust the
result.

## How it works

1. **Seeded.** One run seed drives every random source, and the seed is written into every trace.
   `seed-check` proves a run repeats before any sim/real comparison is made.
2. **Measured trust.** Gates G0–G6 check real data on its way into sim, in order: integrity, contract,
   timing, plausibility, distributions, dynamics (real actions replayed in sim), and outcomes.
3. **Robots define the twin.** `calibrate` fits each robot's versioned, hashed profile from its traces:
   physics, wear age, per-joint wear and timing. Production loads only approved profiles. `fleet-fit`
   replaces the placeholder wear curves.
4. **Realistic stress, one axis at a time:**

   | Axis | What is modelled |
   |---|---|
   | Wear | per joint and per unit, batched on the GPU |
   | Faults | a catalogue of component failures |
   | Timing | measured latency, jitter and missed ticks |
   | Sensor streams | per-topic rate, delay, drops, freeze and bias |
   | Treadmill | trips, slips, split belts, incline, pushes |

   Results come with confidence intervals, compared on the same scenarios.
5. **Failures become tests.** A robot failure is reproduced in the twin with `groot-twin repro`. If the twin
   can't reproduce it, that becomes a ticket against the twin; if it can, the case joins the regression suite.
   Watchdog trips on hardware flow straight into repro.
6. **Hardware in steps:**
   1. offline shadow;
   2. live shadow;
   3. mirror (all or some joints, with a watchdog);
   4. the rig on hardware, mixing real and fake ROS 2 publishers on the real clock.

## Evidence it works (reference twins)

**Timing.** Robot timing from a synthetic trace (~110 ms latency on a 50 Hz loop) dropped the arm
controller's success from **100% to 1%** on the same 100 scenarios (p = 3e-30). Lockstep sim hid this.

**Calibration.** On a synthetic worn robot, calibration cut replay error from **1.23 to 0.13**. In the bridge,
the profiled twin tracked the robot at nRMSE **0.14 with no watchdog trips**, against 1.23 and 5 trips for a
new-unit twin.

**Treadmill.** The walking controller kept walking after **59% of trips [36, 78]**, and after every push, slip,
split, speed step and incline step. Replaying the recorded belt cut replay error from **0.60 to 0.0001**.

**Streams.** The rig showed commands drifting by RMS **0.23** under realistic stream delays and faults, while
success held at 30/30. The controller was degraded but not yet failing.

## Release decision

A checkpoint ships when all seven checks hold:

1. Its configs pass `seed-check`.
2. The fidelity gates pass on the newest real data.
3. The robot's approved profile is current.
4. Success meets target on the **lower** confidence bound and is not worse than the last release on paired
   scenarios.
5. Robustness drops are fixed or documented.
6. Every past failure in the regression suite still passes.
7. Hardware shadow and mirror sessions are clean, or every trip is explained and reproduced in the twin.

## Honest limits

- The CPU twins are reduced-order. Final numbers come from Isaac Lab and the robot.
- The ROS 2 transports and the Isaac Lab paths are written but not yet run against hardware here. Lines that
  depend on the setup are marked `VERIFY`.
- The watchdog guards the rig; the robot's e-stop and safety controller stay in charge.

More: [test philosophy](test-philosophy.md) · [test plan](gr00t-test-plan.md) · each doc's **Limits** section.
