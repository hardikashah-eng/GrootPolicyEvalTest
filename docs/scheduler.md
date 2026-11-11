# Multi-rate scheduling with the robot's timing

Sim loops usually run the policy, the controller and physics in lockstep: the
action computed from the observation at tick *k* acts at tick *k*, every tick
happens exactly on time, and nothing is ever missed. A robot's loops do not
wait for each other. `groot_eval.scheduler` runs the sim on the robot's
timing instead, measured from the robot's own traces.

## 1. Measure the robot's timing

```bash
groot-robot timing-profile traces/real --out profiles/timing.yaml --during continue
```

| Field | Measured from |
|---|---|
| `control_hz` | the trace's fps |
| `latency_ms` | `inference_latency_ms`; with `chunk_index`, the largest value per chunk (one inference per chunk) |
| `control_jitter_ms` | diffs of `timestamp` (the robot clock) minus the nominal period |
| `missed_tick_rate` | gaps of about *k* periods count as *k - 1* missed ticks |
| `policy_period_ticks` | median control ticks between chunk starts |
| `during_inference` | `hold` (blocking deploy script) or `continue` (asynchronous executor); not observable from the trace, so you set it |

Distributions are stored as 101 quantiles, so the YAML stays small and
sampling reproduces the measured tails, not just the mean. The profile records
its source trace, episode count and sample counts.

## 2. Does the policy survive it? (CPU)

```bash
groot-robot timing-check --profile profiles/timing.yaml --env arm3 --policy arm-ik --episodes 100 --out runs/timing
```

Runs the same scenarios twice, lockstep and with the robot's timing, and
reports both success rates with Wilson intervals and a paired McNemar test.
On a synthetic trace with about 110 ms inference latency on a 50 Hz loop and
3.6% missed ticks, the reference `arm-ik` controller drops from 100% to 1%
(99 of 100 scenarios lost, p = 3e-30): it was tuned for a loop that never
waits.

On the CPU path, `ScheduledEnv` holds the previous command on missed or late
control ticks, and `TimingProfile.wrap_policy` adds inference latency through
the existing `RealTimePolicy`.

## 3. Batched, for Isaac Lab

```python
from groot_eval.scheduler import ChunkBuffer, MultiRateScheduler, TimingProfile

timing = TimingProfile.load("profiles/timing.yaml")
sched = MultiRateScheduler(timing, num_envs=env.num_envs, physics_dt=env.physics_dt)
chunks = ChunkBuffer(timing.during_inference)

for _ in range(steps):                      # one iteration per physics step
    tick = sched.step()
    if tick.request.any():                  # inference starts from this observation
        chunks.request(tick.request, policy.predict_chunk(obs))   # (E, H, A)
    chunks.ready(tick.ready)                # results whose latency has elapsed
    if tick.control.any():
        cmd = chunks.command(tick.control & ~tick.missed)  # missed ticks keep the last command
        env.set_joint_targets(cmd, env_mask=tick.control)
    obs = env.physics_step()
```

Every environment draws its own latencies, jitter and missed ticks.
`ChunkBuffer` is array-API code, so the chunks can stay on the GPU; the
scheduler's masks are small NumPy arrays.

## Limits

* The batched path has not been run inside Isaac Lab yet (no GPU here); it is
  tested on NumPy and against the strict array-API namespace.
* `policy_period_ticks` measured from a trace includes the effect of latency
  (a new chunk cannot start before the last inference returns), so it is an
  upper bound on the configured inference rate.
* Jitter and missed ticks are drawn independently per tick; bursts of missed
  ticks (a stalled bus) are better modelled with the `comm_stall` fault.
