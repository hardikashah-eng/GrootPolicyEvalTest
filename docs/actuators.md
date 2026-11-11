# Wear- and fault-aware actuators

`groot_eval.actuators` models the drive of every joint in a batch of
environments, so one rollout can hold a whole fleet: every environment is a
different unit at a different age, with its own fault schedule.

```text
command ──► bus (comm_stall, latency_spike)
        ──► motor model (PD with backlash dead band, or a learned MLP actuator net)
        ──► wear (gain loss, friction, stiction)
        ──► limits (torque-limit loss, thermal derating, power_sag, DC torque-speed curve)
        ──► actuator faults (motor_off, motor_runaway, joint_lock) ──► applied effort
joint position ──► encoder wear (scale, bias, bias walk, noise) + sensor faults ──► reading
```

| Piece | Where |
|---|---|
| Per-env, per-joint wear from an age profile (`WearConfig`) or a measured robot profile | `WearTable.fleet`, `WearTable.from_config`, `WearTable.from_robot_profile` |
| Per-env fault windows from a fault catalogue (`FaultConfig`) or explicit schedules | `FaultTable.sample`, `FaultTable.from_schedules` |
| The batched model | `WearAwareActuatorCore.compute` / `.observe` |
| Learned actuator network (`.npz`, Isaac Lab `ActuatorNetMLP` style) | `MLPMotorModel` |
| Torque-speed curve | `DCMotorSaturation` |
| Isaac Lab actuator class | `groot_eval.actuators.isaaclab_actuator.WearAwareActuatorCfg` |

Wear draws are identical to `WearConfig.sample`: unit `s` is the same unit at
every age, so age comparisons are paired. `typical_units=True` gives the
fleet-median unit (no spread, signed biases at zero).

## One code path for CPU and GPU

The core is written against the Python array API standard
(`array_api_compat`): the same object accepts NumPy arrays or torch tensors
and returns the same type on the same device. The tests run it on NumPy and on
`array_api_strict`, which rejects anything outside the standard, so torch
compatibility does not depend on NumPy-only behaviour. Sensor noise is drawn
with NumPy and copied to the input's device.

## Quick check of a profile

```bash
groot-robot actuator-sweep --wear configs/wear/arm3_profiles_extended.yaml \
  --ages 0,2920,8760 --units 64 --joints 3 --out runs/actuator_sweep
```

Tracks a sine on a gravity-loaded link for every unit at every age in one
batch and reports tracking error by age (paired against the same units when
new), torque saturation and motor temperature. Add `--faults` to give each
unit a fault schedule.

## Isaac Lab

```python
from groot_eval.actuators.isaaclab_actuator import WearAwareActuatorCfg

robot.actuators["legs"] = WearAwareActuatorCfg(
    joint_names_expr=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"],
    effort_limit=88.0, stiffness=150.0, damping=5.0,
    wear_config="configs/wear/arm3_profiles_extended.yaml", ages_hours=[0, 2920, 8760],
    fault_config="configs/faults/arm3_single.yaml", horizon_steps=1000,
)
```

Environments are split evenly across `ages_hours`. Resets restart the fault
clock and cool the motors; wear stays, because it is the unit's age.

## Limits

* **Not yet run inside Isaac Lab** (no GPU where this was written). The
  adapter follows the Isaac Lab 3.0 `ActuatorBase` API; lines likely to need
  adjusting are marked `VERIFY`.
* `joint_lock` is a stiff brake, not a kinematic lock; set the simulator's
  joint drive for an exact lock.
* Backlash is modelled as a dead band on the position error, not as a
  separate motor-side state.
* The wear numbers in `configs/wear/` are placeholders until fitted to your
  robots.
