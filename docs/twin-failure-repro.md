# Digital-twin failure reproduction and conversion fidelity

Judge a digital twin by whether it leads to the same decisions as the real robot, not by how real it looks. This guide covers three tools built on that idea:

* `groot-twin repro`: takes failed runs (from the robot or a higher-fidelity sim) and tries to make the twin fail the same way. It scores the twin on them and, when the twin misses a failure, searches the twin's fidelity knobs for the setting that explains it.
* `groot-twin metrics`: computes tier-1 usefulness metrics from paired sim and real results.
* `groot-twin convert-check`: checks that converting HDF5 demonstrations (Isaac Lab Mimic, robomimic) to LeRobot v2, the format GR00T trains on, lost no visual or actuator information.

Code: [`twin_metrics.py`](../src/groot_eval/twin_metrics.py), [`repro.py`](../src/groot_eval/repro.py), [`conversion_fidelity.py`](../src/groot_eval/conversion_fidelity.py), [`twin_cli.py`](../src/groot_eval/twin_cli.py). The agent skill that drives the loop end to end is [`.claude/skills/twin-failure-repro/SKILL.md`](../.claude/skills/twin-failure-repro/SKILL.md).

## The metrics

### Tier 1: does the twin predict reality?

These need paired results: the same policies on the same tasks, in the twin and on the robot.

| Metric | Definition | Reading |
|---|---|---|
| Pearson r | Correlation of per-policy success rates, twin against robot | Scores move together. Unreliable below about 5 policies of clearly different quality. |
| Spearman / Kendall tau | Rank correlation; tau is the share of policy pairs ordered the same way, minus those ordered the other way | Same ranking of checkpoints. |
| **MMRV** | mean over i of max over j of \|R_i − R_j\| · 1[(S_i − S_j)(R_i − R_j) < 0] (SIMPLER) | Worst mis-ranking, in success-rate units. 0 means the twin never orders two policies the wrong way. For checkpoint selection this matters more than r. |
| Calibration | Bias = mean(sim − real) with a bootstrap CI; least-squares fit real ≈ slope · sim + offset | A known, constant bias is usable. A slope far from 1 means the twin stretches or compresses differences between policies. |
| Sensitivity alignment | For each policy and perturbation, Δ = rate(perturbed) − rate(nominal) in each domain. Reports the correlation of the twin's Δ against the robot's Δ, sign agreement and mean \|error\| | Whether a lighting, object or distractor change hurts as much in the twin as on the robot. A twin can match nominal success and still miss this. |
| Decision agreement | Recall, precision and false-alarm rate of "block" decisions made from twin results against robot results | Whether gating on the twin makes the robot's calls. This is the number leadership asks about. |

### Tier 2: where is it wrong?

Measure each part of the twin separately.

| Metric | Definition | Reading |
|---|---|---|
| Replay nRMSE | Recorded actions are played through the twin from the recorded start state. Per-signal RMS error divided by the recorded signal's spread | Dynamics error with the policy out of the loop. |
| Divergence step | First step where any signal's error exceeds 0.5 of its spread | Early divergence points at gain, mass, units or delay. Late divergence points at friction or damping. |
| DTW distance | Dynamic-time-warping distance per aligned step, in units of the signal's spread | Small DTW with large nRMSE means a timing problem (latency, delay), not a different path. |
| Actuator tracking | Per joint: tracking nRMSE, the lag (in steps) that best aligns measured with commanded, and gain | Compare twin and robot. A lag of 3 on the robot and 0 in the twin is a missing actuator delay. |
| Saturation fraction | Share of steps at the torque or command limit | A real robot saturating where the twin doesn't means a weak or worn motor. |
| Latency match | Wasserstein-1 distance and KS statistic between twin and robot inference latencies, also expressed in control steps | Whether the twin feels the robot's latency. |
| Failure-mode confusion | Matrix of original failure modes against the twin's outcomes, with mode agreement and the JS divergence of the two mode mixes | A twin can match the success rate while failing for different reasons. |
| **Failure reproduction rate** | Share of real failures the twin reproduces: on the same seed, at least `min_same_mode_rate` of twin runs fail with the same mode. Reported with a Wilson interval, at the baseline twin and with the knob search allowed | The single most useful twin metric. Every failure it can't reproduce is a ticket. |
| Policy-relevant visual gap | Embedding cosine distance and action divergence on matched real and twin frames, using the policy's own encoder and action head | The visual gap that matters to the policy. Pixel metrics alone overstate or understate it. |

### Tier 3: is it worth it?

| Metric | Definition | Reading |
|---|---|---|
| Control-variate savings | rho = correlation of per-scenario twin and real outcomes; variance factor 1 − rho²; each real run is worth 1 / (1 − rho²) runs | The share of real rollouts the twin saves at the same confidence, using SureSim-style estimation. |
| Coverage against the field | Share of fleet scenario parameters inside the twin's tested envelope | Low coverage means the twin is validated on conditions the robot doesn't live in. |

## Failure reproduction (`groot-twin repro`)

```bash
# failures recorded on the robot (LeRobot trace with success=false episodes)
groot-twin repro traces/real --config configs/repro/reach2d_example.yaml --domain real --out runs/repro
# failures from a sweep or a regression suite
groot-twin repro runs/candidate --policy pd --env reach2d --out runs/repro_sweep
groot-twin repro runs/candidate/suite.json --policy pd --knob action_delay=0,1,2,4 --out runs/repro_suite
```

Each failure goes through four stages:

1. **Open-loop replay** (when states and actions were recorded): the twin starts in the recorded state and plays the recorded actions. The output is per-signal nRMSE, the divergence step, DTW, and how the twin ended the episode.
2. **Closed-loop re-runs**: the same policy runs `repeats` times on the failing seed, then on `seed_jitter` neighbouring seeds with the same parameters. The case is judged on its own seed. The neighbours show whether the failure is a knife edge or a region.
3. **Verdict**: `reproduced`, `partial`, `not_reproduced` or `flaky`. Flaky means identical re-runs disagree, and that has to be fixed first.
4. **Knob search** (when not reproduced): every setting of the configured knobs is scored. When there is a recording, the ranking is the same-mode rate on the seed minus the replay nRMSE, because the replay is direct evidence about dynamics and a wrong physics setting can still hit the failure mode by accident. The best setting that reproduces the failure becomes a **twin ticket**, e.g. "twin reproduces this failure only with action_delay=5". Two notes flag weak results: a best value at the edge of its range (widen the range), and ties (the knob isn't pinned down by this failure).

On the CPU test, the "robot" was Reach2D with hidden actuator gain 3.0 and 5 steps of delay. The baseline twin reproduced none of the failures. The knob search reproduced all of them and recovered gain 3.0 and delay 5 exactly, with replay nRMSE 0.000 at that setting.

Before acting on a twin ticket, confirm it on clean recordings with `groot-eval replay --config` (system identification). Tuning the twin to one failure is overfitting until the same setting also fits the robot's normal runs.

Knob names must be scenario parameters the twin env accepts, plus `latency_mode` (any `--latency-mode` value). For Isaac Lab, map them through the adapter's `param_setters`. See `configs/repro/isaaclab_template.yaml`.

## Conversion fidelity (`groot-twin convert-check`)

```bash
groot-twin convert-check --hdf5 demos.hdf5 --lerobot datasets/my_task --config configs/conversion/mapping_example.yaml --out runs/conv
# reference converter, to separate pipeline bugs from unavoidable codec loss
groot-twin convert --hdf5 demos.hdf5 --config configs/conversion/mapping_example.yaml --fps 30 --out datasets/ref
```

| Gate | Checks |
|---|---|
| C0 inventory | Episodes and lengths match. Every HDF5 dataset is either converted or listed under `ignore`; anything else is reported as lost, and force/torque, depth, tactile or extra cameras are called out by name. Episode attributes survive. |
| C1 timing | Control rate is kept, nothing was resampled, timestamps are regular, each video has one frame per row, and no stream is shifted by a step. |
| C2 actuator | Per state and action dimension: error relative to the signal's spread, SNR in dB, and **effective bits**, log2(range / (RMS error · √12)), the resolution of an ideal quantizer that would leave the same error. It also detects permuted joints, flipped signs, degrees vs radians and other scales, constant offsets, quantization, clipping, high-frequency loss (filtering), velocity mismatch and new NaNs. |
| C3 visual | Per camera: resolution, RGB/BGR order, float [0, 1] frames cast to black, frame alignment (dropped first frame), PSNR (mean and p5), SSIM, sharpness kept, colour shift, frozen frames, and optionally the distance in the policy's embedding space (`--encoder module:fn`). |
| C4 metadata | Task strings and language instructions, success labels, and whether modality.json slices and order and info.json shapes match the data. |

Exit codes are the same as `fidelity`: 0 pass, 1 fail, 2 a required gate couldn't run. Thresholds live in the YAML file. They're starting points to tune on your own data, not validated limits.

**Codec measurements on the synthetic test content** (80×64 pixels, sharp edges, which is a hard case):

| Codec | Mean PSNR |
|---|---|
| Near-lossless H.264 (`h264-lossless`, the reference converter's default) | Infinite (exact round trip) |
| H.264 CRF 18, yuv420p | About 33 dB |
| AV1 CRF 30, yuv420p (LeRobot's own default encoder settings) | About 32 dB (29 dB p5), with some frozen motion |
| OpenCV mp4v | About 32 dB |

Real camera content compresses differently, so measure your own data with `--codec av1` against `h264-lossless`.

**AV1 decoding.** OpenCV builds often can't decode AV1, and AV1 is LeRobot's default codec. The checker falls back to the `ffmpeg` command line when OpenCV returns no frames. `open-loop --video` and `import-real --video` still use OpenCV only.

## Status and limits

* **Tested here:** everything on a CPU, on Reach2D and synthetic HDF5 data (`tests/test_twin_metrics.py`, `tests/test_repro.py`, `tests/test_conversion_fidelity.py`).
* **Not run here:** Isaac Lab twins and live GR00T servers, because there's no GPU. The repro pipeline only calls `make_env`, `make_policy`, `run_episode` and the optional `set_state`, so any env the harness already runs works.
* **Unknown scenarios:** when a robot recording carries no scenario (seed and parameters), the twin uses the episode index as the seed and default parameters, and the report says so. Record `scenario` in `TraceRecorder.start_episode(...)` to make closed-loop reproduction meaningful.
* **Tier-1 metrics need several policies.** With fewer than 5 policies on both sides, or real rates spanning under 10 points, the report warns instead of pretending.
