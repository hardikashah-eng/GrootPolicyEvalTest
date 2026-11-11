---
name: twin-failure-repro
description: Take a failed robot or sim run, reproduce it in the digital twin, and score the twin with failure-reproduction, replay, failure-mode and tier-1 metrics. Also checks that HDF5 to LeRobot conversion lost no visual or actuator information. Use for "reproduce this failure in sim", "why doesn't the twin fail like the robot", "is our digital twin good enough", or "did the HDF5 to LeRobot conversion lose anything".
---

# Twin failure reproduction

You are evaluating a digital twin by asking whether it fails the way the robot (or a higher-fidelity sim) failed. Every step runs the `groot-twin` and `groot-eval` CLIs in this repo, and the numbers come from their reports. Never claim a failure was reproduced, or a conversion was lossless, without quoting the report's numbers. Metric definitions are in `docs/twin-failure-repro.md`; read it if a number in a report is unclear.

## Inputs to pin down first

Ask only for what you can't find in the repo or the run directory:

1. **Failed runs**: a LeRobot trace or recording (episodes with `success: false`), a sweep run directory (`episodes.jsonl`), a regression suite (`suite.json`), or a failure-record JSON (`{"trace": ..., "episode_index": ..., "failure_reason": ...}`).
2. **Twin env**: `reach2d`, `arm3`, `isaaclab:<Task>` or `module:Class`, plus `--env-arg` values.
3. **Policy that failed**: `groot://host:port` for a GR00T server, or a registered policy.
4. **Data provenance**: whether the failed data was converted from HDF5. If it was, find the HDF5 source and the mapping.

Setup: `uv sync` (or `pip install -e '.[hdf5]'`). Confirm `groot-twin -h` works.

## Workflow

### 1. Gate the data before blaming the twin

- If the failure data came through an HDF5 to LeRobot conversion, run:
  `groot-twin convert-check --hdf5 <src.hdf5> --lerobot <dataset> --config <mapping.yaml> --out runs/conv`
  - Exit 1 means information was lost or changed. Stop and report the failing gate (C0 to C4) with its numbers. Common causes: degrees vs radians, a joint-order swap, a one-step shift, resampling, BGR order, downscaling, a dropped first frame, low PSNR or SSIM.
  - If the cause isn't clear, rerun the source through the reference converter (`groot-twin convert ... --codec h264-lossless`) and check that output. If the reference passes and the pipeline fails, the bug is in the pipeline. If both lose quality, it's the codec: compare `--codec av1` with `h264-lossless`.
- Run the contract check between the failure trace and the twin: `groot-eval check --reference <failed trace> --env <twin> --policy <policy> --out runs/check.md`. Fix contract errors (units, key order, rates) before anything else, because every later number would measure the bug.

### 2. Baseline reproduction

`groot-twin repro <failed> --env <twin> --policy <policy> --domain real --out runs/repro_baseline`

Read `runs/repro_baseline/repro.md`. For each case, note:

- the verdict;
- the same-seed and neighbouring-seed same-mode rates;
- the replay nRMSE, the divergence step, and the worst signals.

| Verdict | What to do |
|---|---|
| `flaky` | Identical re-runs disagree. Report it, then make the twin deterministic for a fixed seed (sensor noise, async physics, a policy sampling temperature) before going further. |
| `reproduced` | Record it. If neighbouring seeds don't fail, call it a knife-edge failure. |
| `partial` or `not_reproduced` | Go to step 3. |

### 3. Hypothesis search over twin knobs

Choose knobs from the evidence, not at random:

| Evidence | Knobs to try |
|---|---|
| Replay error already large in the first steps | `actuator_gain`, `mass`, units |
| Error that grows slowly | `friction`, damping |
| Small DTW with large nRMSE | `action_delay`, `latency_mode` |
| Failure mode `collision` or `missed_grasp` | `action_delay`, `latency_mode`, `sensor_noise`, camera offset |
| Failure mode `drop` or `slip` | `friction`, `mass`, gripper force |
| Failure mode `timeout` | `friction`, `actuator_gain` |

`repro.KNOB_HINTS` holds the same table, and the report suggests knobs when none are configured.

Run, for example:
`groot-twin repro <failed> --env <twin> --policy <policy> --knob action_delay=0,1,2,3,5 --knob actuator_gain=0.6,1.0,1.5,2.0 --out runs/repro_knobs`

Then iterate, at most 3 rounds, keeping the grid under about 64 settings:

- A note saying a knob's best value is at the edge of its range: widen that range.
- A note saying settings tie: that knob isn't pinned down by these failures. Say so; don't pick one.
- Knob names must be scenario parameters the env accepts, or `latency_mode`. For Isaac Lab, they come from the adapter's `param_setters`.

### 4. Confirm the ticket, don't overfit

A twin ticket ("reproduces only with action_delay=5") is a hypothesis. Before recommending a change to the twin's defaults:

- Run `groot-eval replay <clean robot recordings> --env <twin> --config <grid around the ticket>`. The same setting should also fit the robot's normal runs.
- Re-run step 2 with the ticket applied (`--env-arg` or fixed params) and check that previously reproduced failures still reproduce.

### 5. Score the twin

- Failure reproduction rate with its interval, at baseline and with knobs, from `repro.md`.
- Failure-mode confusion and JS divergence.
- If paired sim and real results exist (CSV with `policy, domain, condition, success` and optionally `scenario_id`), run `groot-twin metrics <csv> --gate <bar> --out runs/twin_metrics`. It reports Pearson, Kendall, MMRV, calibration bias and slope, sensitivity alignment, decision agreement and control-variate savings. Repeat its warnings: fewer than 5 policies, or real rates spanning under 10 points, means the correlation numbers are weak.

### 6. Report

Write a short report in the repo's report style: findings first, numbers with intervals, no adjectives. Include:

1. Data gate result: conversion C0 to C4 and the contract check.
2. A per-failure table: original mode, twin outcome, verdict, replay nRMSE, divergence step, and the ticket if any.
3. The twin's scores: reproduction rate with its CI at baseline and with knobs, mode agreement, and tier-1 metrics if run.
4. Twin tickets, each with its evidence and whether it was confirmed in step 4.
5. Open questions: failures not reproduced by any knob, and which knob family to add next. Name the knob, e.g. "contact stiffness is not exposed by the env".

## Guardrails

- Don't change the twin's defaults or commit configs on the strength of one failure. Propose the change with its evidence.
- If a recording has no scenario (seed and parameters), closed-loop results are indicative only; the report notes this. Recommend recording `scenario` in `TraceRecorder.start_episode(...)`.
- Isaac Lab twins and live GR00T servers need a GPU. If none is available, say so and run only the replay-free and data checks you can.
- Commit nothing and push nothing unless the user asks. Follow the repo's PR conventions when they do.
