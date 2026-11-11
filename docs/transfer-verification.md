# Transfer verification: what crosses the sim/real boundary intact

`groot-eval transfer` checks data (or a policy) moving **sim → real** or **real → sim**
for three properties, plus checks specific to each direction:

* **Fidelity**: nothing was lost on the way (precision, range, bandwidth, columns, metadata).
* **Compatibility**: the receiving side reads it the same way (layout, units, conventions).
* **Usability**: the receiving side can actually use it (training, the target env, the policy).

It complements the data-fidelity gates (`groot-eval fidelity`, G0–G6), which ask whether sim
and real describe the *same world* (timing, physical plausibility, distribution, dynamics,
outcomes). Run `fidelity` to judge the sim; run `transfer` every time data or a checkpoint
crosses over.

```bash
# sim data / a sim-validated policy going to the robot
groot-eval transfer --direction sim2real --sim traces/sim --real traces/real \
  --robot configs/robots/arm3.yaml --env arm3 --policy arm-ik \
  --wear configs/wear/arm3_profiles.yaml --out runs/transfer

# robot logs going into sim (calibration, digital twin, replay seeds)
groot-eval transfer --direction real2sim --sim traces/sim --real traces/real \
  --robot configs/robots/arm3.yaml --env arm3 --out runs/transfer_r2s

# a conversion step (export, import, resample, re-encode): verify it is lossless
groot-eval transfer --roundtrip traces/real traces/real_converted --out runs/roundtrip

# real to sim for wear: how old does this robot behave?
groot-eval wear-fit traces/real --env arm3 --wear configs/wear/arm3_profiles.yaml --out runs/wear_fit
```

Exit codes: 0 pass, 1 a gate failed, 2 a required gate could not run. Thresholds:
`configs/transfer/defaults.yaml`. Robot limits: `configs/robots/*.yaml`. All defaults are
starting points to tune on your own data, not validated limits.

## Gates

| Gate | Checks | Fails on |
|---|---|---|
| T0 round trip (`--roundtrip`) | same episodes before/after a conversion: every value, length, column, task string, metadata key, provenance field, fps, modality | any value changed by more than 1e-4 of the signal's spread, NaN introduced, length/episode/task/fps/layout change (dropped columns and metadata warn) |
| T1 compatibility | schema version, state keys and their **order** inside `observation.state`, action dims and group order, control rate, dtypes (float16), camera keys, embodiment / robot type, calibration id, declared units, **swapped joints** (distribution matching), **flipped action signs** and actuator gain (what each command does to the state), quaternion order and norm | any layout, order, rate, id or convention mismatch; T2 and T4 are then skipped |
| T2 fidelity | per signal on the promoted side: NaN, **quantization** (distinct levels vs the reference), **clipping** (samples piled at min/max), **high-frequency content** (filtering, resampling, missing noise), columns that are missing or all-NaN, metadata coverage (success labels, tasks, scenario names) | NaN, coarse quantization, clipping, missing timestamps |
| T3 usability | `modality.json` present, episodes at least one action chunk long, task strings, dead action channels, duplicate episodes, success labels; with `--env`: keys, widths, action size, real states inside what the sim can represent, `set_state` for replay; with `--policy`: finite, right-sized actions on the promoted observations and open-loop error vs the reference | missing modality, all episodes too short, dead action channels, env cannot represent the data, policy fails or returns NaN / wrong size |
| T4 direction | see below | see below |

## What each direction can verify

### Sim → real (sim data or a sim-validated policy goes to the robot)

| Question | How it is verified | Command / gate |
|---|---|---|
| Does sim data stay inside the robot's joint, velocity and action limits? | sim states/actions vs the robot profile | `transfer` T4 `robot.limits`, `robot.action_range` |
| Can the actuators follow the commands? | per-step command change vs `max_action_rate` | T4 `robot.action_rate` |
| Does the sim model the robot's inference latency? | p95 latency in control steps, sim vs real | T4 `latency`; fix with `--latency-mode trace:` |
| Is the sim as noisy and as jerky as the robot? | robust noise estimate per signal; p95 command change | T4 `noise`, `jerk` (warn) |
| Does the rendered camera match the real one? | resolution vs the robot profile; brightness/contrast/sharpness distance | T4 `camera`; `fidelity` G4 image rows |
| Did the sim cover the deployed robot's wear? | robot's `robot_age_hours` vs the `wear.age_hours` in the sim data | T4 `wear` (names the profile to run) |
| Same action post-processing (clip, filter, chunk mode)? | `action_postprocess` in both traces' `trace_info.json` | T4 `postprocess` |
| Same keys, order, units, signs, rate? | T1 | `transfer` T1, `check`, `fidelity` G1 |
| Does sim cover the states the robot visits? | share of real states inside the sim envelope | `fidelity` G4 coverage |
| Does sim rank checkpoints like the robot? | paired outcomes | `fidelity` G6, `correlate` |
| Does a policy accept the robot's observations? | policy on real observations; open-loop error ratio | `transfer --direction real2sim --policy` |

### Real → sim (robot logs go into sim: calibration, digital twins, replay seeds)

| Question | How it is verified | Command / gate |
|---|---|---|
| Which calibration produced these joint angles? | `calibration_id` recorded; equal on both sides | T4 `calibration`, T1 `calibration_id` |
| Zero offsets applied once, not twice? Encoder wrap? | real states inside the robot's limits and inside what the sim can represent | T4 `robot.limits`, T3 `env.range` |
| Are the recorded actions the applied ones? | `action_source: applied`; actions inside the driver's range | T4 `actions`, `robot.action_range` |
| Can the sim represent every recorded signal? | env observation keys and widths vs the trace | T3 `env.keys`, `env.width` |
| Can real episodes seed replays? | env has `set_state` | T3 `env.set_state`; `replay`, `fidelity` G5 |
| Does the sim's reset distribution cover the robot's start states? | real first states inside the sim's first-state envelope | T4 `start_states` |
| Is the robot's latency recorded so the sim can replay it? | `inference_latency_ms` present | T4 `latency` |
| Can sim and real episodes be paired? | scenario names on real episodes | T4 `pairing`; `pairs`, `correlate` |
| How worn is this robot, and what wear should its twin get? | replay the trace through `WearEnv(sim)` at each age, pick the best fit | `wear-fit` |
| Does the sim move like the robot given the same commands? | open-loop replay, optionally after system identification | `replay`, `fidelity` G5 |
| Are success labels trustworthy? | label vs final task reward | `fidelity` G3 `labels` |

### Both directions

Lossless conversions (T0), layout and conventions (T1), information preserved (T2), usable by
training, the env and the policy (T3), plus the integrity, timing and plausibility gates in
`fidelity` (G0, G2, G3).

## What cannot be verified offline

* Contact-rich dynamics the sim does not model at all (deformables, cable drag): replay error
  shows *that* the sim is wrong, not what is missing.
* Safety under conditions no trace covers: coverage (G4) says where data is missing, not what
  would happen there.
* Signed sensor biases and noise from replay (`wear-fit` sets biases to zero and cannot fit
  noise); use `gap` offsets and spreads.
* Anything the trace does not record: provenance (`calibration_id`, `robot_age_hours`,
  `action_source`, `action_postprocess`) must be written into `trace_info.json` by the robot's
  logger for those checks to run.

The example numbers in tests come from the synthetic Arm3 model, not a real robot.
