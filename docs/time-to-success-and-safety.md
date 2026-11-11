# Time to success, safety metrics and paired comparisons

Most real-robot VLA evaluations report binary success at a fixed timeout, from 25 or fewer rollouts, without intervals. That throws away most of what a run measured:
- Two policies with the same 80% can differ 2x in how long they take.
- A policy can "succeed" while slamming into joint limits.
- A timeout is not a fast run, and it should not be dropped either.

`groot-eval dist` scores runs as distributions.

Code: [`distributional.py`](../src/groot_eval/distributional.py). Tests: [`test_distributional.py`](../tests/test_distributional.py).

## Time to success

Failures are **censored**: the robot had not succeeded by the time the episode ended.

| Number | Meaning |
|---|---|
| Kaplan-Meier curve | Share of episodes not yet successful at each time |
| Median, p25, p75, p90 time to success | Read off the curve; "not reached" when fewer than half succeed |
| RMST | Restricted mean time to success up to a horizon: the average time spent before succeeding, where a failure costs the whole horizon. Lower is better. Bootstrap interval. |

## Safety, from traces

| Metric | From |
|---|---|
| `jerk_rms` | Third difference of the commanded action |
| `max_jump` | Largest change of any command dimension between two steps; chunk boundaries show up here |
| `near_limit_share` | Share of steps within 5% of a state limit (`--limits` YAML with `lower:`/`upper:`) |
| `saturation_share` | Share of command values at the edge of `--action-range` |
| any `--metric NAME` | The episode's peak of a recorded per-step column, e.g. `contact_force` |

Each metric is reported as median, p95 and max, plus the share of episodes over its limit with a Wilson interval. Set limits with `--threshold max_jump=0.3`, `--threshold max_contact_force=40`.

## Paired comparison

Two runs on the same scenarios, joined on `scenario_id`, for example:
- lockstep vs measured timing;
- two checkpoints;
- a new vs a worn robot.

It reports:
- McNemar on success;
- the RMST difference with a paired bootstrap interval;
- the KS distance between the time-to-success distributions;
- a verdict: B worse, B better, or no significant difference.

## Commands

```bash
# one run
groot-eval dist runs/ckpt_a --dt 0.05 --trace runs/ckpt_a/trace --action-range=-1,1 --out runs/dist_a

# lockstep vs the robot's measured timing; fail CI if B is worse or a safety limit is exceeded
groot-eval dist runs/lockstep --vs runs/measured_timing --dt 0.05 \
    --trace runs/lockstep/trace --trace-b runs/measured_timing/trace \
    --threshold max_jump=0.3 --gate --out runs/dist_timing
```

`--dt` is the control period. Without it, times are in steps.

## Where it fits

- It complements `groot-robot timing-check` (scheduler.md): timing-check asks whether success drops under the robot's real timing; `dist` also asks whether the robot gets slower or jerkier.
- It complements the rig ([rig.md](rig.md)), whose runs it can score the same way.
- It follows the distributional methodology argued for in recent real-robot benchmarks (PhAIL, arXiv 2605.29710).
