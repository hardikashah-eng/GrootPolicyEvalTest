# Real-to-sim calibration demo (SYNTHETIC)

> **SYNTHETIC.** The "real" measurements are generated with planted errors (see `examples/run_real2sim_demo.py`). This shows what each check finds and that the estimates recover known values. It is not a result about a real cell.

| Check | Before | After applying corrections |
|---|---|---|
| R1_scene | fail | pass |
| R2_kinematics | fail | pass |
| R3_cameras | fail | pass |
| R4_placement | fail | fail |
| R5_physics | fail | fail |

## Recovered values

| Quantity | True (planted) | Estimated |
|---|---|---|
| joint zero offsets (deg) | [0.0, 1.146, -0.573] | [0.013, 1.129, -0.58] |
| base shift (mm) | [3.0, -2.0] | [2.98, -2.01] |
| camera correction | 15 mm, 2.00 deg | 15.8 mm, 2.03 deg |
| cube mass | 0.21 | 0.209 ± 0.001 |
| cube friction | 0.45 | 0.450 ± 0.002 |
| cube static_friction | 0.55 | 0.544 ± 0.008 |
| cube restitution | 0.35 | 0.350 ± 0.001 |

After the corrections, what still fails is not a sim parameter:

- `R4_placement` `cube.paired_position`: real start position vs the sim's for the same scenario: mean 7.0 mm, worst 37.4 mm over 10 scenarios (limit 20.0 mm)
- `R4_placement` `cube.support`: measured object centre height vs resting on the surface: worst 12.0 mm, median +0.0 mm (positive = floating, negative = sunk; limit 5.0 mm); fix the pose source before importing these placements into the sim
- `R4_placement` `overlap`: 1 scenario(s) with interpenetrating objects: ['cell_07: cube/cup']
- `R5_physics` `cube.sim_rollout`: your simulator, run with these parameters, reproduces the recorded tests to 0.201 nRMSE (worst; limit 0.15): the engine does not behave like the measured object even with matching parameters (friction combine mode, contact stiffness, solver substeps)
