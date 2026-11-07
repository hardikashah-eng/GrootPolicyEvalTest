# Data-fidelity gates: FAIL

Direction `real2sim`: sim `demo-output/robustness/traces/sim`, real `demo-output/robustness/traces/robot_bad_units`. Gates run in order; after a failure the later gates are skipped, because they would measure the bug instead of the data.

| Gate | Status | Required | Key numbers |
|---|---|---|---|
| G0_integrity | pass (warnings) | yes | 0 errors, 1 warnings |
| G1_contract | **FAIL** | yes | 3 errors, 0 warnings |
| G2_timing | skipped | yes |  |
| G3_plausibility | skipped | yes | 0 errors, 1 warnings |
| G4_distribution | skipped | yes |  |
| G5_dynamics | skipped | yes | 0 errors, 1 warnings |
| G6_outcome | skipped | no | 0 errors, 1 warnings |

## G0_integrity: pass (warnings)

- warn `real.manifest`: no meta/manifest.json; run `groot-eval fidelity --write-manifest DIR` when the data is produced, so later changes are detectable
- ok `sim.manifest`: 64 files match the SHA-256 manifest
- ok `sim.integrity`: 60 episodes, 4510 steps, all finite

## G1_contract: **FAIL**

- **ERROR** `state.units`: state.joint_vel[0]: spread is 0.0214x that of real; looks like degrees vs radians (x57.3, any remainder is likely dynamics)
- **ERROR** `state.units`: state.joint_vel[1]: spread is 0.0213x that of real; looks like degrees vs radians (x57.3, any remainder is likely dynamics)
- **ERROR** `state.units`: state.joint_vel[2]: spread is 0.0203x that of real; looks like degrees vs radians (x57.3, any remainder is likely dynamics)
- ok `state.keys`: same 6 state keys
- ok `fps`: both at 50 Hz

## G2_timing: skipped

- warn `skipped`: contract errors: units, keys or rates differ, so every later number would be wrong

## G3_plausibility: skipped

- warn `skipped`: contract errors: units, keys or rates differ, so every later number would be wrong

## G4_distribution: skipped

- warn `skipped`: contract errors: units, keys or rates differ, so every later number would be wrong

## G5_dynamics: skipped

- warn `skipped`: contract errors: units, keys or rates differ, so every later number would be wrong

## G6_outcome: skipped

- warn `skipped`: contract errors: units, keys or rates differ, so every later number would be wrong
