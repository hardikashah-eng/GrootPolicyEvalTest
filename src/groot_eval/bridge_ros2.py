"""ROS 2 transport for the hardware bridge (needs ``rclpy``; not imported unless used).

Topics (all names configurable):

* ``state_topic`` (``sensor_msgs/JointState``): the robot's joint state. It is
  reordered by ``joint_names`` into ``state.joint_pos``, ``state.joint_vel``
  and ``state.joint_torque`` (``effort``). Each new message is one bridge tick.
* ``applied_topic`` (``std_msgs/Float64MultiArray``): the command the robot's
  controller actually applied, in action order. Shadow mode needs it; in
  mirror mode it reports how the robot merged the bridge's partial command
  with its own (without it, the bridge assumes the robot applied exactly what
  it was sent).
* ``command_topic`` (``std_msgs/Float64MultiArray``): mirror-mode commands.
  ``layout.dim[0].label`` carries the comma-separated indices the bridge
  drives when ``command_mask`` is set, so a robot-side node can merge them
  into its own command.
* ``stop_topic`` (``std_msgs/String``): published on a watchdog trip as
  ``"<hold|stop>: <reason>"``. A robot-side node must act on it; the bridge
  stops sending commands either way.

The bridge does not replace the robot's safety controller or e-stop.

Example ``ros2.yaml``::

    joint_names: [left_shoulder_pitch, left_shoulder_roll, left_elbow]
    state_topic: /joint_states
    applied_topic: /groot/applied_command
    command_topic: /groot/bridge_command
    stop_topic: /groot/bridge_stop
    qos_depth: 10

Lines marked ``VERIFY`` depend on the robot's ROS 2 setup and have not been
run against a robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from groot_eval.bridge import RobotSample
from groot_eval.types import Scenario


@dataclass
class Ros2Config:
    joint_names: list[str]
    state_topic: str = "/joint_states"
    applied_topic: str | None = "/groot/applied_command"
    command_topic: str = "/groot/bridge_command"
    stop_topic: str = "/groot/bridge_stop"
    node_name: str = "groot_bridge"
    qos_depth: int = 10
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> Ros2Config:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not data.get("joint_names"):
            raise ValueError(f"{path}: joint_names is required (the order of the state and action vectors)")
        return cls(**data)


def joint_state_to_obs(names: list[str], msg_names: list[str], position: Any, velocity: Any, effort: Any,
                       order: list[str]) -> dict[str, np.ndarray]:
    """Reorder one ``JointState`` into bridge observations (pure function, so it is testable without ROS)."""
    index = {n: i for i, n in enumerate(msg_names)}
    missing = [n for n in order if n not in index]
    if missing:
        raise KeyError(f"JointState has no {missing}")
    idx = [index[n] for n in order]
    obs = {}
    for key, values in (("state.joint_pos", position), ("state.joint_vel", velocity), ("state.joint_torque", effort)):
        v = np.asarray(list(values), dtype=np.float64)
        if v.size == len(msg_names):
            obs[key] = v[idx].astype(np.float32)
    if "state.joint_pos" not in obs:
        raise ValueError("JointState carries no positions")
    return obs


class Ros2Robot:
    supports_mirror = True

    def __init__(self, cfg: Ros2Config) -> None:
        import rclpy  # VERIFY: ROS 2 environment sourced (Humble or newer)
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray, String

        self.cfg = cfg
        self._Float64MultiArray, self._String = Float64MultiArray, String
        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self.node: Any = Node(cfg.node_name)
        self._state: Any = None
        self._fresh = False
        self._applied: np.ndarray | None = None
        self.node.create_subscription(JointState, cfg.state_topic, self._on_state, cfg.qos_depth)
        if cfg.applied_topic:
            self.node.create_subscription(Float64MultiArray, cfg.applied_topic, self._on_applied, cfg.qos_depth)
        self.pub_cmd = self.node.create_publisher(Float64MultiArray, cfg.command_topic, cfg.qos_depth)
        self.pub_stop = self.node.create_publisher(String, cfg.stop_topic, cfg.qos_depth)
        self._stopped = False

    @classmethod
    def from_config(cls, path: str | Path) -> Ros2Robot:
        return cls(Ros2Config.load(path))

    def _on_state(self, msg: Any) -> None:
        self._state, self._fresh = msg, True

    def _on_applied(self, msg: Any) -> None:
        self._applied = np.asarray(msg.data, dtype=float)

    def _wait(self, timeout_s: float) -> RobotSample | None:
        import time

        end = time.monotonic() + timeout_s
        while not self._fresh and time.monotonic() < end:
            self._rclpy.spin_once(self.node, timeout_sec=max(0.0, end - time.monotonic()))
        if not self._fresh:
            return None
        self._fresh = False
        m = self._state
        obs = joint_state_to_obs(self.cfg.joint_names, list(m.name), m.position, m.velocity, m.effort,
                                 self.cfg.joint_names)
        stamp = float(m.header.stamp.sec) + 1e-9 * float(m.header.stamp.nanosec)  # VERIFY: robot fills header.stamp
        applied = None if self._applied is None else self._applied.copy()  # latest applied command
        return RobotSample(stamp, obs, applied)

    def start(self, scenario: Scenario) -> RobotSample:
        self._stopped = False
        first = self._wait(5.0)
        if first is None:
            raise TimeoutError(f"no JointState on {self.cfg.state_topic} within 5 s")
        first.command = None
        return first

    def tick(self, command: np.ndarray | None, mask: np.ndarray | None, timeout_s: float) -> RobotSample | None:
        if command is not None and not self._stopped:
            msg = self._Float64MultiArray()
            msg.data = [float(x) for x in np.asarray(command, dtype=float).reshape(-1)]
            if mask is not None:
                from std_msgs.msg import MultiArrayDimension

                dim = MultiArrayDimension()
                dim.label = ",".join(str(i) for i in np.flatnonzero(mask))
                dim.size = len(msg.data)
                dim.stride = len(msg.data)
                msg.layout.dim = [dim]
            self.pub_cmd.publish(msg)
        sample = self._wait(timeout_s)
        if sample is not None and sample.command is None and command is not None:
            sample.command = np.asarray(command, dtype=float)  # no applied_topic: assume it applied what we sent
        return sample

    def safe_stop(self, reason: str, how: str) -> None:
        self._stopped = True
        msg = self._String()
        msg.data = f"{how}: {reason}"
        self.pub_stop.publish(msg)

    def close(self) -> None:
        self.node.destroy_node()
