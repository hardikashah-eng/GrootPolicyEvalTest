# Hardware bridge: the twin next to the robot

`groot-robot bridge` runs the digital twin and the real robot side by side,
one control tick at a time. It compares them tick by tick, and a watchdog
reacts when they come apart.

> The watchdog is a test-rig guard, not a safety system. Keep the robot's
> e-stop, joint limits and safety controller in charge, and start every
> mirror session with the robot in a safe pose and someone at the stop button.

## Modes

| Mode | Who commands the robot | The twin gets | Use it to |
|---|---|---|---|
| `shadow` | the robot's own controller | the command the robot applied | check the twin against a running robot without touching it |
| `mirror` | the bridge: a policy, or recorded commands | the same command | run a policy, or replay captured data, on the hardware with the twin in lockstep |

With `command_mask: [..]`, mirror mode drives only some action dimensions
(one arm of a humanoid, say). The robot's controller keeps the rest, and
reports the merged command it applied, which is what the twin gets.

Each tick goes like this:

1. Read the robot's state and the command it applied.
2. Step the twin with that same command.
3. Compare the twin's next state with the robot's.
4. Run the watchdog.

The twin starts from the robot's state (`set_state`). With `resync_every: N`
it is put back into the robot's state every N ticks, so the report can show
error by steps since sync, like `groot-eval replay`.

## Watchdog

| Check | Trips when |
|---|---|
| `divergence` | the twin-robot error on a signal in `max_error` stays over its limit for `consecutive` ticks |
| `missing` | the robot sends nothing for `max_missing` ticks in a row |
| `stale` | the robot clock jumps by more than `max_stale_s` |
| `state_limit` | a robot signal leaves its `state_limits` range |
| `command_step` | a mirror command changes by more than `max_command_step` in one tick |
| `non_finite` | a command or a robot state contains NaN or inf |

What a trip does depends on the mode:

- **Mirror:** the bridge calls the transport's `safe_stop` (`on_trip: hold` or `stop`), stops sending, and ends the session.
- **Shadow:** the bridge records the first trip and keeps watching (`stop_on_trip: true` ends the episode instead).

See `configs/bridge/shadow.yaml` for every setting.

## Robots

| `--robot` | What it is |
|---|---|
| `sim:<env>[:k=v,...]` | an env standing in for the robot. Give it other physics than the twin to rehearse a gap: `sim:arm3:friction=1.8,action_delay=3`. Its own controller is `--robot-policy`. |
| `trace:<dir>` | a recorded robot trace, played back as the robot stream (offline shadow; no commands) |
| `ros2:<config.yaml>` | ROS 2. Uses `sensor_msgs/JointState` in, `std_msgs/Float64MultiArray` commands out, and the robot's applied command on a second topic. See `configs/bridge/ros2.yaml` and `groot_eval/bridge_ros2.py`. Needs `rclpy`. |

In mirror mode, `--source` sets where the commands come from:

- `policy:<name>`: a policy acting on the robot's observations.
- `trace:<dir>`: recorded actions replayed open loop, one recorded episode per bridge episode, with the scenario taken from the trace.

## Examples

```bash
# Rehearse with a stand-in robot that has more friction than the twin
groot-robot bridge --robot sim:arm3:friction=1.8 --robot-policy arm-ik --twin arm3 --episodes 3 --out runs/shadow

# Offline: a recorded robot session against the twin
groot-robot bridge --robot trace:traces/real --twin arm3 --out runs/offline

# On the robot: replay captured data, arm joints only, stop on divergence
groot-robot bridge --mode mirror --robot ros2:configs/bridge/ros2.yaml --source trace:traces/real \
    --twin arm3 --config configs/bridge/mirror_arm.yaml --out runs/mirror
```

## Output

| File | Contents |
|---|---|
| `bridge.md`, `bridge.json` | trips; per-signal nRMSE and max error; error by steps since sync; every 10th tick plus every tick within 5 of a trip; G5; repro hand-off |
| `steps_episode<N>.csv` | every tick: robot clock, steps since sync, worst signal, max abs error per signal |
| `robot/`, `twin/` | both sides as traces (`real` and `sim`). Robot steps carry `bridge.err.<signal>` columns. A tripped episode ends with `failure_reason: watchdog:<reason>`. |
| `repro_cases.json` | one failure record per tripped episode, in the format `groot-twin repro` reads |

The report reuses the existing tools rather than adding new metrics:

- **Live comparison:** scored like `replay` (nRMSE over the robot's spread; signals that do not move are left out).
- **G5 dynamics** (`groot_eval.fidelity`): replays the robot's recorded commands in a fresh twin, with the default gate limits.
- **Twin reproduction:** `--repro-policy <name>` runs `groot-twin repro` on the tripped episodes and adds `repro.md`. Without it, the report prints the command to run.

## Limits

- The ROS 2 transport has not been run against a robot here (no ROS 2 or
  hardware in this environment). Its message handling is unit-tested
  (`joint_state_to_obs`); lines that depend on the robot's setup are marked
  `VERIFY`.
- `stop_topic` only asks for a stop: a robot-side node must act on it.
- The bridge compares the signals both sides publish under the same name.
  Over ROS 2 those are `joint_pos`, `joint_vel` and `joint_torque` (`effort`),
  so the twin must report the same units and joint order.
- The loop is paced by the robot's samples. The bridge adds the twin step and
  the comparison to each tick, so keep the twin faster than real time.
