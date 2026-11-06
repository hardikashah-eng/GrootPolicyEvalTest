> **SYNTHETIC.** The "real" trace is the built-in Reach2D simulator with hidden physics: friction 1.0, actuator gain 0.9, 2 steps of actuation delay, sensor noise 0.03 (policy `pd-chunk`, 4 steps per chunk, 20 episodes), relabelled `domain: real`. Candidates come from `configs/reach2d_sysid.yaml`. It shows the report format and that the fit recovers known values, not a real robot result.

# Replay: recorded actions through the simulator

- Trace: `real_trace` (domain: real), 20 episodes, 1263 steps
- Env: `reach2d`; started from recorded state: yes
- Pure open loop: never re-synced after the first step

nRMSE = RMS(sim state - recorded state) / std(recorded state). 0 is perfect; around 1 means the sim is about as wrong as always predicting the mean. Signals that never change within an episode (a target set at reset) are not scored.

## 1. Baseline: (env defaults)

Score (mean nRMSE): **0.976**; 20 episode(s) ended early in sim

| Signal | nRMSE |
|---|---|
| position | 0.855 |
| velocity | 1.097 |

Error growth (mean |error| / std, by steps since the sim was last put in the recorded state):

| Steps | 1 | 5 | 10 | 20 |
|---|---|---|---|---|
| error | 0.327 | 0.678 | 0.879 | 0.979 |

Error at step 1 that is already large points at a one-step model error (gain, mass, units, a missing delay); error that starts small and grows points at slow drift (friction, damping).

## 2. Parameter fit (144 settings of `friction`, `actuator_gain`, `action_delay`)

| Rank | Score | position | velocity | Parameters |
|---|---|---|---|---|
| baseline | 0.976 | 0.855 | 1.097 | (env defaults) |
| #1 | 0.052 | 0.104 | 0.000 | friction=1, actuator_gain=0.9, action_delay=2 |
| #2 | 0.141 | 0.119 | 0.164 | friction=1, actuator_gain=0.9, action_delay=1 |
| #3 | 0.144 | 0.124 | 0.163 | friction=1, actuator_gain=0.9, action_delay=3 |
| #4 | 0.146 | 0.163 | 0.129 | friction=1.2, actuator_gain=1, action_delay=2 |
| #5 | 0.156 | 0.176 | 0.136 | friction=1.2, actuator_gain=1, action_delay=3 |
| #6 | 0.158 | 0.130 | 0.185 | friction=1.2, actuator_gain=1.1, action_delay=3 |
| #7 | 0.163 | 0.129 | 0.197 | friction=1.4, actuator_gain=1.2, action_delay=3 |
| #8 | 0.169 | 0.159 | 0.179 | friction=0.8, actuator_gain=0.8, action_delay=2 |
| #9 | 0.170 | 0.174 | 0.165 | friction=0.8, actuator_gain=0.8, action_delay=1 |
| #10 | 0.170 | 0.160 | 0.180 | friction=1.2, actuator_gain=1.1, action_delay=2 |

- best setting cuts tracking error by 95% versus the baseline (0.976 -> 0.052)
- 4 replayed episode(s) ended (terminated) before the recording did

### Which knob matters

Best score reachable with each value of a parameter (others free). A large spread means the recording constrains that parameter; a flat row means it does not.

| Parameter | Spread | Best score by value |
|---|---|---|
| friction | 0.656 | 1: 0.052, 1.2: 0.146, 1.4: 0.163, 0.8: 0.169, 0.6: 0.356, 0.4: 0.708 |
| action_delay | 0.160 | 2: 0.052, 1: 0.141, 3: 0.144, 0: 0.212 |
| actuator_gain | 0.132 | 0.9: 0.052, 1: 0.146, 1.1: 0.158, 1.2: 0.163, 0.8: 0.169, 0.7: 0.184 |
