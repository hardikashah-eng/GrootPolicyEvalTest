"""ROS 2 side of the test rig (needs ``rclpy``; imported only when used).

Two pieces connect the rig in :mod:`groot_eval.rig` to a real ROS 2 graph:

* :class:`Ros2RigPublisher` - the rig as a set of **fake publishers**. Every message the rig delivers goes out
  on ``<prefix>/<topic>`` as it arrives, on the real clock (run the rig with ``clock: real``). Header stamps are
  the sample times, in wall time. A controller under test (a ros2_control stack on mock or real hardware,
  or a GR00T node) subscribes to these topics instead of the drivers'.
* :class:`Ros2LiveSource` - a **live topic as a rig source**: the latest message of a real sensor (a camera,
  an IMU, joint states), so real inputs can be mixed with replayed and synthetic ones, per topic.

Message types: ``state.*`` arrays use ``std_msgs/Float64MultiArray``; ``video.*`` frames use
``sensor_msgs/Image`` (rgb8). Do not publish ``/clock`` or set ``use_sim_time`` when anything drives real
hardware: the rig runs on wall time.

Lines marked ``VERIFY`` depend on the ROS 2 install and have not been run against a robot here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from groot_eval.rig import Source


def array_payload(value: Any) -> list[float]:
    """A ``state.*`` value as Float64MultiArray data (pure, testable without ROS)."""
    return [float(x) for x in np.asarray(value, dtype=float).reshape(-1)]


def image_payload(frame: Any) -> dict[str, Any]:
    """An RGB frame as sensor_msgs/Image fields (pure, testable without ROS)."""
    arr = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 RGB frame, got shape {arr.shape}")
    h, w, _ = arr.shape
    return {"height": h, "width": w, "encoding": "rgb8", "is_bigendian": 0, "step": 3 * w, "data": arr.tobytes()}


def topic_name(prefix: str, key: str) -> str:
    return f"{prefix.rstrip('/')}/{key.replace('.', '/')}"


def _stamp(node: Any, t_wall: float) -> Any:
    from builtin_interfaces.msg import Time  # VERIFY

    sec = int(t_wall)
    return Time(sec=sec, nanosec=int((t_wall - sec) * 1e9))


class Ros2RigPublisher:
    """Publishes every message the rig delivers, as it is delivered (hook it into RigSession via ``on_arrive``)."""

    def __init__(self, keys: list[str], prefix: str = "/rig", node_name: str = "groot_rig", qos_depth: int = 10) -> None:
        import rclpy  # VERIFY: ROS 2 environment sourced
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from std_msgs.msg import Float64MultiArray

        if not rclpy.ok():
            rclpy.init()
        self.node: Any = Node(node_name)
        self._Image, self._Array = Image, Float64MultiArray
        self.pubs = {k: self.node.create_publisher(Image if k.startswith("video.") else Float64MultiArray,
                                                   topic_name(prefix, k), qos_depth) for k in keys}

    def __call__(self, key: str, value: Any, sampled_wall: float) -> None:
        if key.startswith("video."):
            msg = self._Image(**image_payload(value))
            msg.header.stamp = _stamp(self.node, sampled_wall)
        else:
            msg = self._Array()
            msg.data = array_payload(value)
        self.pubs[key].publish(msg)

    def close(self) -> None:
        self.node.destroy_node()


class Ros2LiveSource(Source):
    """The latest message on a live ROS 2 topic, as a rig source (``source: {kind: live, fn: ...}`` wrapper)."""

    def __init__(self, topic: str, msg_type: str = "array", field: str = "data", node: Any = None) -> None:
        import rclpy  # VERIFY
        from rclpy.node import Node

        if not rclpy.ok():
            rclpy.init()
        self._rclpy = rclpy
        self.node = node or Node(f"groot_rig_live_{abs(hash(topic)) % 10000}")
        self.latest: Any = None
        if msg_type == "image":
            from sensor_msgs.msg import Image as T
        elif msg_type == "joint_state":
            from sensor_msgs.msg import JointState as T
        else:
            from std_msgs.msg import Float64MultiArray as T
        self.msg_type, self.field = msg_type, field
        self.node.create_subscription(T, topic, self._on_msg, 10)

    def _on_msg(self, msg: Any) -> None:
        if self.msg_type == "image":
            self.latest = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, -1)
        else:
            self.latest = np.asarray(getattr(msg, self.field), dtype=float)

    def value(self, t: float, plant_obs: Any) -> Any:
        self._rclpy.spin_once(self.node, timeout_sec=0.0)
        if self.latest is None:
            raise RuntimeError("no message received yet on the live topic")
        return self.latest
