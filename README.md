# groot-policy-eval

**An evaluation harness for robot foundation-model policies such as NVIDIA Isaac GR00T.**

Robot policies don't pass or fail a test the way code does. The same checkpoint succeeds on one seed and drops the cup on the next, and it can look great in simulation and fall short on the robot. This harness treats evaluation as a measurement problem:

* **Closed-loop sweeps** run a policy through hundreds of randomized episodes (friction, mass, actuator gain, sensor noise, latency, obstacles) and report success rates **with confidence intervals**, not bare percentages.
* **Failure attribution** shows which randomized parameter actually drives failure, and only flags it when the evidence is statistically significant.
* **Checkpoint comparison** says whether a new checkpoint is significantly better or worse than the baseline, and returns a non-zero exit code on regression so it can gate CI.
* **Regression suites** save every failing scenario (exact seed and parameters) and replay them against the next checkpoint: what got fixed, what still fails, and which previously passing anchors broke.
* **Open-loop evaluation** scores predicted action chunks against recorded demonstrations in **LeRobot v2 format** (the format GR00T fine-tuning uses), including how error grows along the action chunk.
* **Sim-to-real correlation** checks whether simulation ranks checkpoints the same way the real robot does, and which scenarios disagree most.
* **Sim-to-real gap debugging** logs sim and real runs step by step in one format, checks the two sides agree on units, ranges, rates and key order, ranks every signal by how far sim and real drift apart, and replays real actions in sim to find the physics parameters that reproduce the robot (see [below](#debugging-the-sim-to-real-gap)).
* **Wear and tear by age:** the same scenarios on a brand-new, 1-week, 1-year-old robot (motor gain and torque loss, joint friction, backlash, stiction, encoder bias and noise, camera blur, bus latency), compared pair by pair, plus a **degree-of-freedom loss** matrix (each joint locked, limp or range-limited). See [below](#robustness-wear-faults-skill-suites-and-data-fidelity-gates).
* **Fault injection:** sensors, motors, joints, gripper and comm bus fail mid-episode, one at a time or several at random over long tasks. Reports say whether the robot finished, stopped safely or did something unsafe, and how fast it reacted.
* **Skill-focused suites:** one skill, its conditions at min / nominal / max, run one at a time, at every corner and as a Latin hypercube, so interactions show up that random sweeps miss.
* **Data-fidelity gates (G0–G6):** integrity, contract, timing, physical plausibility, distribution and coverage, dynamics replay and outcome agreement, in order, with thresholds in YAML and CI exit codes, for data moving sim to real or real to sim.
* **Adapters** for a running **Isaac-GR00T inference server** and for **Isaac Lab** tasks, plus built-in CPU simulators (a point mass and a 3-joint arm with a gripper) so the whole workflow runs on a laptop.

## Quick start

```bash
uv sync
uv run python examples/run_demo.py      # every step below, under a minute on a CPU
uv run python examples/run_sim2real_demo.py   # the sim-to-real loop against a synthetic robot, ~15 s
uv run python examples/run_robustness_demo.py # aging, DOF loss, faults, skill suite, fidelity gates, ~2 min
uv run pytest                           # 102 tests
```

The reports from that run are checked in under [`docs/sample-reports/`](docs/sample-reports/).

## What the demo shows

The built-in **Reach2D** simulator is small on purpose, but it has the properties that make real robot evaluation hard: randomized physics, sensor noise, actuation delay, an obstacle, and several distinct failure modes. Two reference controllers stand in for two checkpoints: a naive one, and one that steers around obstacles.

**1. A sweep with intervals** ([report](docs/sample-reports/closed_loop_candidate.md))

```text
groot-eval sweep --policy pd-avoid --config configs/reach2d_sweep.yaml --out runs/candidate
Success 73.3% [68.1, 78.0] (220/300); failures: {'collision': 77, 'timeout': 3}
```

**2. What drives failure.** The obstacle and actuation delay are flagged as significant. Actuator gain, sensor noise, mass and friction are not, even though some raw gaps look large, because their buckets' intervals overlap:

| Parameter | Gap | Significant | Buckets |
|---|---|---|---|
| obstacle | 67.2% | **yes** | False: 98.4% (n=188); True: 31.2% (n=112) |
| action_delay | 36.0% | **yes** | 0: 92.8%; 1: 79.2%; 2: 66.2%; 4: 56.8% |
| actuator_gain | 14.7% | no | 0.60–0.76: 81.3%; … ; 0.90–1.06: 66.7% |

**3. Is the new checkpoint better?** ([report](docs/sample-reports/compare.md)) 62.3% → 73.3%, z = 2.88: a significant improvement. Swap the arguments and the command exits 1, which is what a CI gate needs.

**4. Regression suite** ([report](docs/sample-reports/regression.md)): of the baseline's 113 failing scenarios, the candidate fixes 33 and every anchor scenario still passes, so the gate passes. The 80 still failing are almost all obstacle episodes with actuation delay. With the obstacle present, the candidate succeeds in 15 of 20 episodes with no delay but only 3 of 32 with four steps of delay. The avoidance logic reacts to where the robot was, not where it will be, which points at the next fix.

**5. Action chunking.** Executing 16 steps of each predicted chunk before re-planning scores 58.3% [52.7, 63.8]; re-planning every 2 steps scores 62.0% [56.4, 67.3]. The intervals overlap, so **this sweep can't say chunk length matters overall**, even though the 16-step version adds a new failure mode (`out_of_bounds`, 10 episodes). `groot-eval plan` says how many episodes it would take to resolve a gap that size.

**6. Open-loop** ([report](docs/sample-reports/open_loop.md)): action MSE grows from 0.0000 at the first step of the chunk to about 0.03 by step 8. That is the policy's internal model drifting from the real dynamics, and it is a cheap early warning before any closed-loop run.

**7. Sim-to-real** ([report](docs/sample-reports/correlation.md)) on a **synthetic** example: Spearman 0.66 between sim and real checkpoint rankings, 68% outcome agreement on matched scenarios, and a list of the scenarios where sim and real disagree most.

## Commands

| Command | What it does |
|---|---|
| `groot-eval sweep` | Closed-loop evaluation over a randomization config. `--export-suite` saves failures; `--min-success 0.8` fails the command if the CI lower bound is below 0.8. |
| `groot-eval compare BASE CAND` | Two-proportion test between two sweeps; exit 1 on significant regression. |
| `groot-eval regress --suite` | Replay a regression suite; exit 1 if an anchor broke. |
| `groot-eval open-loop --dataset` | Score action chunks against a LeRobot v2 dataset. `--video` decodes camera streams listed in `modality.json`. |
| `groot-eval correlate pairs.csv` | Sim-to-real agreement from paired results (`policy, scenario_id, domain, success`), plus a matching section that says which rows didn't join and why. |
| `groot-eval sweep ... --export-pairs CSV` | Append the sweep's outcomes as `domain=sim` rows for `correlate`; `--pairs-policy` sets the checkpoint label to match the real runs. |
| `groot-eval pairs TRACE --out CSV` | Append a trace's episode outcomes (sim or real) to a pairs CSV. |
| `groot-eval sweep ... --trace DIR` | Also write every step (observations, actions, timing, chunk position) as a LeRobot v2 trace. Works on `regress` too. |
| `groot-eval sweep ... --latency-mode MODE` | Make the sim feel inference latency the way a robot does: `measured`, `fixed:<ms>`, `normal:<mean>,<std>` or `trace:<dir>` (sample a real trace's latencies). `--during-inference hold\|continue` picks what runs while inference is in flight. Works on `regress` too. |
| `groot-eval import-real DATASET --out DIR` | Convert a real-robot LeRobot recording into a trace and validate it (missing columns, NaNs, declared fps vs timestamps, jitter). `--video` adds image statistics. |
| `groot-eval check --reference REAL` | Contract check of an env rollout (`--env`, `--policy`) or a trace (`--traces`) against a reference: keys, dimensions, degrees vs radians, scale and offset, gripper scaling, control rate, image range and resolution, GR00T action-key order. Exit 1 on errors. |
| `groot-eval gap SIM REAL --out DIR` | Sim-to-real gap report: signals ranked by distance, timing and latency, action-chunk boundaries, outcomes. `--plots` adds histograms (needs matplotlib). |
| `groot-eval replay TRACE --out DIR` | Put the sim in each recorded episode's first state, play the recorded actions open loop, and report per-signal tracking error. `--config` ranks every candidate parameter setting by fit (system identification); `--reset-every N` re-syncs every N steps. |
| `groot-eval make-demo-data` | Record a small LeRobot-format dataset from the built-in simulator. |
| `groot-eval aging --wear W.yaml` | Same scenarios at every age profile; paired McNemar test against the new robot, robustness margin, which wear drives failure, `--dof-episodes N` adds the DOF-loss matrix. `--min-success-at year_1 0.8` gates. |
| `groot-eval faults --faults F.yaml` | Fault campaign: success, safe outcome, unsafe events, controlled stops, reaction and recovery, by fault type, component and number of concurrent faults. `--min-safe` / `--max-unsafe` gate. Combine with `--wear --profile`. |
| `groot-eval suite --config S.yaml` | Skill suite: one-at-a-time, corners and LHS over min / nominal / max conditions. `--min-oat` / `--min-corner` fail on cells significantly below the bar. Combine with `--wear`, `--faults`. |
| `groot-eval fidelity --sim S --real R` | Data-fidelity gates G0–G6 (`--gates` YAML, `--direction sim2real\|real2sim`, `--env` for replay, `--pairs` for outcomes). Exit 0 pass, 1 fail, 2 a required gate could not run. `--write-manifest DIR` records SHA-256 hashes. |
| `groot-eval sweep ... --wear W --profile P` / `--faults F` | Any sweep on a worn robot or with faults. |
| `groot-eval plan --margin 0.05` | Episodes needed for a given confidence-interval width. |

Policies: `pd`, `pd-avoid`, `pd-chunk`, `arm-ik`, `arm-ik-robust`, `groot://host:port`, or `package.module:Class`. Environments: `reach2d`, `arm3`, `isaaclab:<TaskId>`, or `package.module:Class`. Constructor options go through `--policy-arg key=value` and `--env-arg key=value`.

## Randomization config

```yaml
seed: 7
episodes: 300
params:
  friction:      {uniform: [0.2, 1.2]}
  mass:          {uniform: [0.5, 2.0]}
  sensor_noise:  {uniform: [0.0, 0.04]}
  action_delay:  {choice: [0, 1, 2, 4]}
  obstacle:      {choice: [false, true], weights: [0.6, 0.4]}
```

`mode: grid` (with `grid_points` and `repeats`) sweeps every combination instead. Scenarios are generated deterministically from the seed, so every episode can be replayed exactly.

## Using it with GR00T

1. Start an Isaac-GR00T inference server for your checkpoint and embodiment (see the Isaac-GR00T README for the exact flags in your version).
2. **Open-loop** on your fine-tuning dataset:
   `groot-eval open-loop --dataset <lerobot_dataset> --policy groot://localhost:5555 --video --horizon 16 --out runs/open_loop`
3. **Closed-loop** in simulation: implement an environment that emits GR00T's observation keys (`video.*`, `state.*`), or use `isaaclab:<Task>` with `param_setters` for the physics you want to randomize, then `groot-eval sweep --env ... --policy groot://localhost:5555`.

The adapter maps observation keys (`obs_map`), concatenates the action keys you name (`action_keys`), adds batch dimensions, and passes the task instruction under `annotation.human.action.task_description`. The client class is looked up from known Isaac-GR00T import paths and can be overridden with `client_cls`.

## Debugging the sim-to-real gap

Success rates say *that* sim and real differ. Finding *why* needs both sides recorded the same way, so they can be compared signal by signal.

**1. Record both sides in one format.** A trace is a LeRobot v2 dataset (what GR00T fine-tuning uses, so `open-loop` reads it unchanged) with extra per-step columns: `timestamp`, `wall_time`, `inference_latency_ms`, `chunk_index`, `chunk_step`, `reward`, and `image.<camera>.{brightness,contrast,sharpness,height,width,max_value}` for each `video.*` key. `meta/trace_info.json` records `domain: sim|real`, the control period and the observation keys. The full schema is in [`trace.py`](src/groot_eval/trace.py).

* Sim: `groot-eval sweep ... --trace traces/sim`.
* Real, from your own control loop: the same recorder, three calls:

  ```python
  from groot_eval.trace import TraceRecorder
  rec = TraceRecorder(domain="real", env="my_cell", policy="groot@robot", control_dt=1 / 30)
  rec.start_episode("pick up the cup")
  # each step: the obs the policy saw, the action it returned, how long inference took
  rec.record_step(obs, action, latency_ms=ms, timestamp=robot_clock_s)
  rec.end_episode(success=True)
  rec.write("traces/real")
  ```
* Real, from an existing LeRobot recording: `groot-eval import-real my_dataset --out traces/real`.

**2. Check the contract before the physics.** Many sim-to-real gaps turn out to be bugs:
* a joint in degrees on one side and radians on the other;
* a float image in [0, 1] that `astype(uint8)` turns black;
* a gripper commanded 0–1 in sim and 0–100 on the robot;
* action groups concatenated in another order;
* data recorded at 30 Hz driving a 20 Hz controller.

`groot-eval check --reference traces/real --env <sim> --policy <ckpt>` looks for each of these. The GR00T adapter now scales [0, 1] float images instead of zeroing them, and warns when `action_keys` isn't set.

**3. Find the gap.** `groot-eval gap traces/sim traces/real --out runs/gap` writes `gap.md` and `gap.json` ([synthetic sample](docs/sample-reports/gap_SYNTHETIC.md)):
* **Signal ranking:** every state dimension, action dimension, per-step action change and image statistic, ranked by Wasserstein distance in pooled standard deviations, each with a KS test and a likely-cause hint.
* **Timing:** control-period jitter and inference latency, also expressed in control steps. Sim waits for the policy, the robot doesn't, so this tells you how much delay sim is missing.
* **Action chunking:** how much the command jumps when a new chunk starts, compared with steps inside a chunk.
* **Outcomes:** success rates with intervals.

In a CPU check, the sim had friction 0.4–0.8, and the stand-in "real" had friction 0.9–1.2, sensor noise and 2 steps of hidden actuation delay. Contract checks came out clean. The chunk-boundary jump ratio was 2.5 on "real" versus 1.3 in sim, which points at the missing delay, and actions were jerkier on "real".

The comparison is between distributions, not matched trajectories. Some differences are behaviour rather than physics, so check observation rows before action rows; step 4 tells the two apart.

**4. Replay real actions in sim, then fit the sim.** `groot-eval replay traces/real --env reach2d --out runs/replay` puts the sim in each real episode's first state, plays the recorded actions open loop and measures how far the simulated states drift (nRMSE per state signal, plus error growth by steps since sync). The policy is out of the loop, so a gap here is the sim's dynamics, not the policy reacting to them. Error that is already large after one step points at gain, mass, units or a missing delay; error that grows slowly points at friction or damping.

Add `--config configs/reach2d_sysid.yaml` (a grid of candidate physics) and every setting is replayed and ranked ([synthetic sample](docs/sample-reports/replay_SYNTHETIC.md)). The report flags a best value at the edge of the searched range, parameters the data doesn't pin down (several settings within 5% of the best), and how much each parameter moves the score. On a stand-in "real" trace with friction 1.0, gain 0.9, 2 steps of delay and sensor noise, the fit picked exactly those values out of 144 settings and cut tracking error by 95%.

Replay needs an optional `set_state(state, recent_actions)` on the env: Reach2D has one; Isaac Lab takes a task-specific `state_setter` (not run here, no GPU). Without it, replay starts from the seed's reset state. Fit noise parameters with `gap`, not replay: noise only adds open-loop error. Recorded actions are what the policy commanded; if the robot driver clips or filters them, record what was applied.

**5. Give the sim the robot's latency.** The closed-loop runner waits for the policy, so in sim a command computed from the observation at tick *k* acts at tick *k*. On the robot it acts at tick *k* + ceil(latency / dt), and something has to run in between. `--latency-mode` on `sweep` and `regress` models that ([`latency.py`](src/groot_eval/latency.py)):
* `--latency-mode trace:traces/real` samples the inference latencies the robot recorded; `fixed:120` or `normal:110,25` work without a recording, and `measured` uses the real call time (for a live GR00T server).
* `--during-inference hold` keeps sending the last command while inference runs (a blocking deploy script). `continue` keeps executing the current chunk and starts the new one at the step that matches the time that passed (an asynchronous executor).
* With zero latency it is exactly the plain policy. The modelled latency is what the trace records, so `gap` compares like with like.

In a CPU check, the stand-in "real" ran `pd-chunk` with about 110 ms of inference latency (2–3 control steps) in `continue` mode. A plain sim sweep differed from it by up to 0.52 σ (velocity) with a chunk-boundary ratio of 1.0 vs 5.3. The same sweep with `--latency-mode trace:<real> --during-inference continue` was within 0.01 σ on every signal, ratio 5.3 vs 5.3. With `hold` instead, actions were 0.57 σ off, so the gap report also tells you which execution mode the robot really uses. This "real" is synthetic and was built with the same latency model, so it shows the mechanism, not that the model fits a particular robot.

**6. Pair sim and real outcomes per scenario.** `correlate` joins on `scenario_id`, but a sweep's ids are hashes of seed and parameters while a robot test set has names like `pick_place_00`, so nothing used to match. Now:
* A sweep config can list **named scenarios** that mirror the real test set (`scenarios: [{name: pick_place_00, seed: 0, params: {...}}, ...]`, with `repeats` runs each). The name doesn't change the scenario's id, so existing suites and results still load.
* `sweep --export-pairs pairs.csv --pairs-policy ckpt_a` appends one `sim` row per episode, keyed by the scenario name.
* Real runs: pass `scenario_name=` to `TraceRecorder.start_episode(...)`, then `groot-eval pairs traces/real --out pairs.csv --policy ckpt_a` appends the `real` rows.
* `correlate` now reports how many policy/scenario pairs matched and what was left out, and warns when one side uses hash ids and the other uses names, or when no checkpoint label appears on both sides.

**The whole loop on a CPU.** [`examples/run_sim2real_demo.py`](examples/run_sim2real_demo.py) runs steps 1–6 against a **synthetic** "robot": Reach2D with hidden friction 1.0, actuator gain 0.9 and 1 step of actuator delay, sensor noise, about 110 ms of inference latency executed asynchronously, and a logger that exports velocity ×57.3. Three checkpoints run on the "robot" (16 named scenarios) and in a naive sim. Results ([summary](docs/sample-reports/sim2real_demo_SYNTHETIC.md)):
* `check` rejects the first export with a degrees-vs-radians error on velocity; after the fix it is clean.
* `gap` on the naive sim: largest gap 0.93 σ, and the robot's 2–3 steps of inference latency are called out.
* `replay` with the parameter grid recovers friction 1.0, gain 0.9 and delay 1 exactly, out of 144 settings. It finds the actuator delay, not the inference latency, because recorded actions already include the latency.
* A calibrated sim (fitted physics plus `--latency-mode trace:` the robot's latencies) brings the largest gap down to 0.12 σ.
* `correlate`: outcome agreement goes from 83% to 98% on 48 matched pairs, and checkpoint ranking agreement from 0.00 to 1.00. The naive sim says the obstacle-avoiding checkpoint never fails. On the "robot" it collides in half the scenarios because it steers on observations that are 2–3 steps old, and only the latency-aware sim reproduces that. The calibrated sim is somewhat pessimistic on the chunked checkpoint (88% vs 100%).

A test runs the demo and fails if the fit stops recovering the hidden values. Everything on the "robot" side is simulated, so this shows the tools work together and find known causes, not that they will on a particular robot.

## Robustness: wear, faults, skill suites and data-fidelity gates

Everything here is a wrapper or a scenario generator, so sweeps, intervals, failure attribution, regression suites and traces work on it unchanged. Wear values and fault schedules are **scenario parameters** (`wear.joint[1].backlash_rad`, `faults: "stuck@encoder[1]:40+60"`), so every degraded episode has an id, is replayed exactly by `regress`, and is ranked by attribution like friction or latency. The CPU examples use **Arm3** (`--env arm3`): a planar 3-joint arm with a gripper, gravity, per-joint motors, gearboxes and encoders. The wear numbers in `configs/wear/` are **placeholders**: calibrate them from durability rigs or fleet logs (`replay --config` fits gain and friction from an old robot's trace). Sample reports from `examples/run_robustness_demo.py` are under [`docs/sample-reports/robustness/`](docs/sample-reports/robustness/) ([summary](docs/sample-reports/robustness/SUMMARY_SYNTHETIC.md)).

**Components.** An env can list what can degrade with `components()` (name, kind, which action indices and state slices it owns) and change its own physics with `set_component_param(name, key, value)`. Arm3 does both; on Isaac Lab, map the same keys to actuator `stiffness`, `damping`, `effort_limit`, `friction` and joint limits through `param_setters`. Envs without these hooks still get the boundary effects: commands are scaled, clipped, zeroed or delayed, and observations are biased, noised, quantized, frozen or dropped. Effects that need physics and have no lever are listed as not applied, not silently skipped.

**1. Wear and tear by age** ([report](docs/sample-reports/robustness/aging.md)). A profile file maps age in hours to each effect (`linear`, `sqrt`, `bathtub`), with a per-unit log-normal spread and per-joint load factors. Each scenario's seed is its unit, so the same unit is compared at every age and the McNemar test on discordant pairs detects smaller drops than two independent sweeps would.

```bash
groot-eval aging --env arm3 --policy arm-ik --config configs/arm3_reach.yaml --episodes 200 \
  --wear configs/wear/arm3_profiles.yaml --dof-episodes 40 --min-success-at year_1 0.9 --out runs/aging
```

On the placeholder profiles: 100% through year 1, 97.5% at year 2, 87.0% at year 3 (26 scenarios lost, p = 3e-8), 78.5% at end of life. The robustness margin at 90% falls between year 2 and year 3. Joint 1 friction is the only significant driver at end of life. The DOF-loss matrix shows a limp shoulder is fatal (0%) while a locked one still allows 47.5% of reaches.

**2. Fault injection** ([robust](docs/sample-reports/robustness/faults_single_robust.md), [naive](docs/sample-reports/robustness/faults_single_naive.md), [long horizon](docs/sample-reports/robustness/faults_long_horizon.md)). Twelve fault types (`stuck`, `dropout`, `bias_step`, `noise_burst`, `blackout`, `motor_off`, `motor_runaway`, `joint_lock`, `gripper_fail`, `comm_stall`, `latency_spike`, `power_sag`) on component globs, one per episode (`mode: single` with a start window) or Poisson arrivals with a concurrency cap (`mode: random`). *Safe outcome* = success or a controlled stop, with no collision, instability or joint-limit hit after the fault. Safety gates must be *shown* to hold (CI lower bound of safe outcomes at or above `--min-safe`), so too few episodes fails the gate.

```bash
groot-eval faults --env arm3 --policy arm-ik-robust --episodes 300 --faults configs/faults/arm3_single.yaml --min-safe 0.9 --out runs/faults
groot-eval faults --env arm3 --env-arg max_steps=900 --policy arm-ik-robust --config configs/arm3_long.yaml \
  --faults configs/faults/arm3_long_horizon.yaml --wear configs/wear/arm3_profiles.yaml --profile year_1 --out runs/faults_long
```

With single faults, the naive controller reaches a safe outcome in 77.8% of episodes and sends NaN commands 27 times when a reading drops out. The robust one (holds the last valid reading, flags a frozen or stalled joint, routes motion through the others, re-trusts the joint after a while) reaches 86.5% with no NaN commands. That still fails a 90% gate, and `motor_off` is why: a limp joint falls onto its limit. In the long task, success is 100% with one fault, 84% with two and 54% with three: the cliff is at the third concurrent fault.

**3. Skill suites** ([report](docs/sample-reports/robustness/suite_reach.md)). Conditions with `min`, `nominal`, `max`; designs `oat` (one at a time), `corners` (2^k, limit with `corner_conditions`) and `lhs` (stratified). Scenario names record the design cell, so `--trace` doubles as a per-skill, per-condition dataset. `configs/suites/` has a runnable Arm3 reach-and-grasp suite and Isaac Lab templates for walking and grasping.

```bash
groot-eval suite --env arm3 --policy arm-ik --config configs/suites/arm3_reach.yaml --min-corner 0.5 --out runs/suite
```

No single condition is significant one at a time (the worst level, two steps of delay, is 83.3% on 12 episodes), but the corners find `friction=min` with `action_delay=max` at 40% and one corner significantly below 50%. A random sweep would have averaged that away.

**4. Data-fidelity gates** ([units bug](docs/sample-reports/robustness/fidelity_units_bug_SYNTHETIC.md), [worn robot](docs/sample-reports/robustness/fidelity_worn_robot_SYNTHETIC.md), [after sysid](docs/sample-reports/robustness/fidelity_after_sysid_SYNTHETIC.md)).

| Gate | Checks |
|---|---|
| G0 integrity | columns, lengths, NaN/Inf, increasing timestamps, SHA-256 manifest (`--write-manifest`), provenance keys |
| G1 contract | the `check` command: keys, dims, units, scale, gripper range, action order, image range, control rate |
| G2 timing | period jitter, dropped steps, `camera_time.<cam>` sync offset |
| G3 plausibility | finite-difference vs reported velocity, position and action limits, teleports, derived signals vs the env's kinematics (`--env`), success labels vs final reward |
| G4 distribution | per-signal Wasserstein distance (state, action, image), coverage of real states by the sim's envelope |
| G5 dynamics | real actions replayed in sim, error at 10 steps, optionally after `--sysid-config` |
| G6 outcome | `correlate` agreement and checkpoint ranking (`--pairs`) |

G0 and G1 are prerequisites: if either fails, the rest are skipped. `--direction` names the side being promoted; problems on the other side only warn. In the demo, a "robot" trace logging joint velocity in deg/s stops at G1. Once fixed, a 3-year-old unit fails G3 (encoder bias puts reported angles past the joint limits), G4 (coverage 92%) and G5 (replay error 1.30). Fitting the sim's global friction brings G5 to 0.49 with the fit pinned at the edge of its range, which G5 flags: per-joint wear is what's missing. A trace compared with itself passes every gate, with a replay error of 0.000. G5 needs the same physics on both sides, so record it with fixed parameters or fit them with `--sysid-config`.

## Status and limits

* **Tested here:** everything that runs on a CPU, including statistics, sweeps, attribution, comparison, regression suites, the LeRobot reader and writer, open-loop scoring, correlation, the CLI, and the GR00T adapter against a fake client.
* **Not run here:** the live GR00T server path and the Isaac Lab adapter, because the development machine has no GPU. Both are written against the published APIs (Isaac-GR00T N1.5 client contract, Isaac Lab 2.x), and both APIs change between releases. Verify key names against your checkpoint's `modality.json` before trusting results.
* The Isaac Lab adapter runs one environment at a time. Batching episodes across Isaac Lab's parallel environments is the obvious next step for throughput.
* Video decoding for open-loop uses OpenCV seeks, which is simple but slow on long episodes.
* The sim-to-real example data is **synthetic**. It shows the input format and the analysis, not a real result.
* Wear profiles and fault rates are **placeholders**; Arm3 is a teaching model, not an arm. Condition coverage of real datasets per suite cell is not implemented yet; G4 coverage works on state space instead.

## Layout

```text
src/groot_eval/
  stats.py           Wilson intervals, bootstrap, two-proportion test, Spearman, sample-size planning
  randomization.py   sweep configs -> reproducible scenarios
  closed_loop.py     episode runner, latency capture, summaries
  analysis.py        failure attribution, worst cases
  regression.py      suite export, replay, determinism check
  open_loop.py       LeRobot v2 reader/writer, chunk scoring
  correlation.py     sim-to-real agreement
  trace.py           step-level traces for sim and real (one LeRobot v2 schema), real-recording import
  checks.py          contract checks: keys, units, ranges, rates, image range, action order
  gap.py             signal-by-signal sim-to-real gap report
  replay.py          open-loop replay of recorded actions, system-identification ranking
  latency.py         inference-latency model for sim rollouts (hold / continue during inference)
  report.py          Markdown reports
  degradation/       wear profiles + WearEnv, fault schedules + FaultEnv, aging / DOF-loss / fault runners and reports
  suites.py          skill suites: min / nominal / max conditions, one-at-a-time, corners, LHS
  fidelity.py        data-fidelity gates G0-G6, SHA-256 manifests
  envs/              Reach2D and Arm3 simulators, Isaac Lab adapter
  policies/          reference controllers (incl. naive and fault-tolerant arm IK), action chunking, GR00T server adapter
```

## License

MIT
