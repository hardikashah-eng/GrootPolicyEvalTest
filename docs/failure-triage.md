# Failure triage: why did this robot run fail?

When a real run fails, the first question is whose problem it is. Forum threads about GR00T deployments often end without an answer for this reason: the policy runs, nothing errors, and the robot does the wrong thing. `groot-twin triage` runs a fixed set of cheap tests on the recorded run and the twin, and names the most likely cause, its evidence, what it ruled out, and the next step.

Code: [`triage.py`](../src/groot_eval/triage.py). Tests: [`test_triage.py`](../tests/test_triage.py), one synthetic failure per cause.

## The tests

| Test | How | Rules in |
|---|---|---|
| T1 policy consistency | Re-query the policy on the recorded observations, in order, and compare with the recorded actions. | Same actions: the deployment ran the policy faithfully. Late actions (a constant offset, or a held command): **timing**, from inference latency. A clean scale, offset, sign flip or reordering: **plumbing** (action mapping, units, normalization, joint order). |
| T2 open-loop replay | Play the recorded actions through the twin from the recorded start state. | Tracks once actions are shifted by k steps: **timing**, from actuation delay. Residual that grows: **actuation** / dynamics. Residual that stays flat and looks like noise: **sensing**. |
| T3 closed-loop twin | Re-run the policy on the same scenario in the twin: same seed, repeats, neighbouring seeds. | Fails the same way: a **policy** failure the twin reproduces. With a T1 timing finding, it re-runs with that latency to confirm it. |

When the policy ran faithfully, the twin tracks the robot, and the twin still does not fail, the verdict is **not_modelled**: something in the real scene is missing from the twin, such as an object, a contact or a disturbance. The next step is `groot-twin repro` with knobs.

## Command

```bash
groot-twin triage traces/real --env arm3 --policy groot://localhost:5555 --out runs/triage
```

The input is the robot's trace, written by `TraceRecorder` or imported with `groot-eval import-real`. Episodes marked failed are triaged. Use `--episodes 3,7` to pick specific ones.

Output:
- `triage.md`: one line per run, then evidence, ruled-out causes and the next step for each.
- `triage.json`: every test's numbers.

Settings such as `max_lag` and `replay_good_nrmse` are in `DEFAULTS` and can be overridden with `--set max_lag=10`.

## Limits

- **T1 needs state-only inputs.** It needs a policy that can act on the recorded `state.*` observations. For camera policies it reports "not checkable", and triage relies on T2 and T3.
- **T2 needs `set_state`.** It needs a twin that can be put into the recorded state; without it, the replay starts from the scenario's reset state. A robot whose first recorded state is outside the twin's valid range (noise near a joint limit) can end the replay early.
- **Confidence is a judgement.** "High" means two tests agree or a re-run confirmed the finding. "Medium" means one test points at the cause and the others do not contradict it. Treat "low" as a lead, not a finding.
- **Thresholds are not yet calibrated.** They were set on the CPU twins, not on robot data.
