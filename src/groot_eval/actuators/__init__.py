"""Wear- and fault-aware actuator models for a whole fleet at once.

:class:`~groot_eval.actuators.core.WearAwareActuatorCore` turns joint commands
into applied efforts for ``num_envs x num_joints`` joints in one call, with a
motor model (PD, DC-motor saturation or a learned network), each environment's
wear (from a :class:`~groot_eval.degradation.WearConfig` age profile or a robot
profile) and each environment's fault schedule (from a
:class:`~groot_eval.degradation.FaultConfig`). Every environment can be a
different unit at a different age, so one batched rollout sweeps a fleet.

The core is written against the Python array API standard: the same code runs
on NumPy (CPU, tested here) and on torch tensors (Isaac Lab, GPU).
:mod:`groot_eval.actuators.isaaclab_actuator` wraps it as an Isaac Lab
``ActuatorBase``.
"""

from groot_eval.actuators.core import (ActuatorParams, FaultTable, WearAwareActuatorCore, WearTable,
                                       wear_at_age)
from groot_eval.actuators.motor_models import DCMotorSaturation, MLPMotorModel

__all__ = ["ActuatorParams", "DCMotorSaturation", "FaultTable", "MLPMotorModel", "WearAwareActuatorCore",
           "WearTable", "wear_at_age"]
