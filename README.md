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
* **Adapters** for a running **Isaac-GR00T inference server** and for **Isaac Lab** tasks, plus a built-in CPU simulator so the whole workflow runs on a laptop.

## Quick start

```bash
uv sync
uv run python examples/run_demo.py      # every step below, under a minute on a CPU
uv run pytest                           # 82 tests
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
| `groot-eval plan --margin 0.05` | Episodes needed for a given confidence-interval width. |

Policies: `pd`, `pd-avoid`, `pd-chunk`, `groot://host:port`, or `package.module:Class`. Environments: `reach2d`, `isaaclab:<TaskId>`, or `package.module:Class`. Constructor options go through `--policy-arg key=value` and `--env-arg key=value`.

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

## Status and limits

* **Tested here:** everything that runs on a CPU, including statistics, sweeps, attribution, comparison, regression suites, the LeRobot reader and writer, open-loop scoring, correlation, the CLI, and the GR00T adapter against a fake client.
* **Not run here:** the live GR00T server path and the Isaac Lab adapter, because the development machine has no GPU. Both are written against the published APIs (Isaac-GR00T N1.5 client contract, Isaac Lab 2.x), and both APIs change between releases. Verify key names against your checkpoint's `modality.json` before trusting results.
* The Isaac Lab adapter runs one environment at a time. Batching episodes across Isaac Lab's parallel environments is the obvious next step for throughput.
* Video decoding for open-loop uses OpenCV seeks, which is simple but slow on long episodes.
* The sim-to-real example data is **synthetic**. It shows the input format and the analysis, not a real result.

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
  envs/              Reach2D simulator, Isaac Lab adapter
  policies/          reference controllers, action chunking, GR00T server adapter
```

## License

MIT
