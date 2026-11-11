"""Isaac Lab adapter: :class:`WearAwareActuator`, an explicit ``ActuatorBase`` backed by the batched core.

Use it like any Isaac Lab actuator in an articulation config::

    from groot_eval.actuators.isaaclab_actuator import WearAwareActuatorCfg

    robot.actuators["legs"] = WearAwareActuatorCfg(
        joint_names_expr=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"],
        effort_limit=88.0, stiffness=150.0, damping=5.0,
        wear_config="configs/wear/arm3_profiles_extended.yaml",
        ages_hours=[0, 2920, 8760],            # environments are split evenly across these ages
        fault_config="configs/faults/arm3_single.yaml", horizon_steps=1000,
        motor_model="profiles/g1-0042/legs_motor.npz",   # optional learned actuator net
    )

Each environment becomes one unit at one age (unit ``env_id % envs_per_age``,
so the same unit appears at every age), with its own fault schedule. Episode
resets restart the step counter and cool the motors for the reset
environments; wear stays (it is the unit's age, not episode state).

STATUS: written against the Isaac Lab 3.0 actuator API (``ActuatorBase``,
``ArticulationActions``) and **not yet run in Isaac Lab** (no GPU here). The
physics core is tested on NumPy and against the strict array-API namespace.
Lines most likely to need adjusting are marked ``VERIFY``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from groot_eval.actuators.core import ActuatorParams, FaultTable, WearAwareActuatorCore, WearTable
from groot_eval.actuators.motor_models import DCMotorSaturation, MLPMotorModel
from groot_eval.degradation.faults import FaultConfig
from groot_eval.degradation.wear import WearConfig

try:  # VERIFY: module paths for Isaac Lab 3.x
    from isaaclab.actuators import ActuatorBase, ActuatorBaseCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.types import ArticulationActions
    HAVE_ISAACLAB = True
except ImportError:  # pragma: no cover - exercised only where Isaac Lab is installed
    HAVE_ISAACLAB = False


def build_core(cfg: Any, num_envs: int, num_joints: int, effort_limit: Any, stiffness: Any, damping: Any,
               robot_wear: dict[str, Any] | None = None) -> WearAwareActuatorCore:
    """Build the batched core from an adapter config (shared by the Isaac Lab class and tests)."""
    if robot_wear is not None:
        wear = WearTable.from_robot_profile(robot_wear, num_envs, num_joints)
    elif cfg.wear_config:
        wcfg = WearConfig.load(cfg.wear_config)
        ages = list(cfg.ages_hours or [0.0])
        per = int(np.ceil(num_envs / len(ages)))
        env_ages = np.repeat(np.asarray(ages, dtype=float), per)[:num_envs]
        seeds = np.arange(num_envs) % per
        wear = WearTable.from_config(wcfg, num_joints, env_ages, seeds, typical=bool(cfg.typical_units))
    else:
        wear = WearTable.new(num_envs, num_joints)
    if cfg.fault_config:
        faults = FaultTable.sample(FaultConfig.load(cfg.fault_config), num_envs, num_joints, int(cfg.horizon_steps),
                                   float(cfg.dt), seeds=np.arange(num_envs) + int(cfg.fault_seed))
    else:
        faults = FaultTable.none(num_envs)
    params = ActuatorParams(effort_limit=effort_limit, stiffness=stiffness, damping=damping,
                            coulomb_friction=cfg.coulomb_friction, viscous_friction=cfg.viscous_friction, dt=float(cfg.dt))
    motor = MLPMotorModel.load(cfg.motor_model) if cfg.motor_model else None
    sat = DCMotorSaturation(cfg.saturation_effort, cfg.velocity_limit_dc) if cfg.saturation_effort else None
    return WearAwareActuatorCore(params, wear, faults, motor=motor, saturation=sat, noise_seed=int(cfg.fault_seed))


if HAVE_ISAACLAB:  # pragma: no cover

    class WearAwareActuator(ActuatorBase):
        """Explicit actuator: computes joint efforts with per-environment wear and faults."""

        is_implicit_model = False

        def __init__(self, cfg: "WearAwareActuatorCfg", *args: Any, **kwargs: Any) -> None:
            super().__init__(cfg, *args, **kwargs)
            robot_wear = None
            if cfg.robot_profile:
                from groot_eval.profiles import RobotProfile  # added by the calibration PR

                robot_wear = RobotProfile.load(cfg.robot_profile).wear_for(self.joint_names)
            self.core = build_core(cfg, self._num_envs, self.num_joints, self.effort_limit[0].tolist(),
                                   self.stiffness[0].tolist(), self.damping[0].tolist(), robot_wear)

        def reset(self, env_ids: Sequence[int]) -> None:
            ids = None if env_ids is None else np.asarray(list(env_ids) if not hasattr(env_ids, "cpu") else env_ids.cpu())
            self.core.reset(ids)

        def compute(self, control_action: ArticulationActions, joint_pos: Any, joint_vel: Any) -> ArticulationActions:
            tau = self.core.compute(control_action.joint_positions, control_action.joint_velocities,
                                    control_action.joint_efforts, joint_pos, joint_vel)
            self.computed_effort = tau  # VERIFY: attribute names in Isaac Lab 3.x
            self.applied_effort = tau
            control_action.joint_efforts = tau
            control_action.joint_positions = None
            control_action.joint_velocities = None
            return control_action

    @configclass
    class WearAwareActuatorCfg(ActuatorBaseCfg):
        class_type: type = WearAwareActuator
        wear_config: str | None = None
        ages_hours: list[float] | None = None
        typical_units: bool = False
        robot_profile: str | None = None
        fault_config: str | None = None
        horizon_steps: int = 1000
        fault_seed: int = 0
        motor_model: str | None = None
        saturation_effort: float | None = None
        velocity_limit_dc: float = float("inf")
        coulomb_friction: float = 0.0
        viscous_friction: float = 0.0
        dt: float = 0.005
