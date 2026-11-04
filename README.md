# groot-policy-eval

**An evaluation harness for robot foundation-model policies such as NVIDIA Isaac GR00T.**

Robot policies don't pass or fail a test the way code does. The same checkpoint succeeds on one seed and drops the cup on the next, and it can look great in simulation and fall short on the robot. This harness treats evaluation as a measurement problem:

* **Closed-loop sweeps** run a policy through hundreds of randomized episodes (friction, mass, actuator gain, sensor noise, latency, obstacles) and report success rates **with confidence intervals**, not bare percentages.
* **Failure attribution** shows which randomized parameter actually drives failure, and only flags it when the evidence is statistically significant.
* **Checkpoint comparison** says whether a new checkpoint is significantly better or worse than the baseline, and returns a non-zero exit code on regression so it can gate CI.
* **Regression suites** save every failing scenario (exact seed and parameters) and replay them against the next checkpoint: what got fixed, what still fails, and which previously passing anchors broke.
* **Open-loop evaluation** scores predicted action chunks against recorded demonstrations in **LeRobot v2 format** (the format GR00T fine-tuning uses), including how error grows along the action chunk.
* **Sim-to-real correlation** checks whether simulation ranks checkpoints the same way the real robot does, and which scenarios disagree most.
* **Adapters** for a running **Isaac-GR00T inference server** and for **Isaac Lab** tasks, plus a built-in CPU simulator so the whole workflow runs on a laptop.

## Quick start

```bash
uv sync
uv run python examples/run_demo.py      # every step below, under a minute on a CPU
uv run pytest                           # 35 tests
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
| `groot-eval correlate pairs.csv` | Sim-to-real agreement from paired results (`policy, scenario_id, domain, success`). |
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
  report.py          Markdown reports
  envs/              Reach2D simulator, Isaac Lab adapter
  policies/          reference controllers, action chunking, GR00T server adapter
```

## License

MIT
