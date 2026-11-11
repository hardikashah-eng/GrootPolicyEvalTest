"""Adapters: Isaac Lab state, or real-room tags and sensors, to the scorer's world state.

Both adapters produce the same world state through :class:`~livechorebench.world.WorldBuilder`, so a task is
scored by the same rules in simulation and in the real room. Only the source of the poses and readings differs:

* :class:`IsaacLabAdapter` reads the simulator: ``env.unwrapped.scene[name].data`` (root pose, joint positions)
  for each object and fixture named in the spec, and the robot's base height.
* :class:`RealRoomAdapter` reads one measurement frame at a time: fiducial tags (AprilTag) or motion capture
  for poses, encoders for joints (a drawer's slide), scales for "is the object in this region", the robot's
  state estimate for base height, and the speech log.

Each object's entry in the spec may name how it is found:

.. code-block:: yaml

    drawer: {usd: ..., real_sku: ..., fixed: true, joint: drawer_slide, real_joint: drawer_encoder}

The real room also needs a **cell file** (surveyed fixture poses, which tag is on which object and its offset,
which scale sits under which region). See ``cells/example_cell.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from livechorebench.spec import TaskSpec
from livechorebench.world import Pose, WorldBuilder, from_frame, quat_from_yaw, quat_to_matrix


def _np(x: Any) -> np.ndarray:
    """Torch tensor (any device) or array-like -> numpy."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x, dtype=float)


class IsaacLabAdapter:
    """World state from an Isaac Lab (or similar) scene.

    ``scene`` maps names to entities with ``data.root_pos_w``, ``data.root_quat_w`` (wxyz), and for articulations
    ``data.joint_pos`` plus ``joint_names``; in Isaac Lab this is ``env.unwrapped.scene``. Arrays are indexed by
    environment first, so one adapter per parallel environment (``env_index``).
    """

    def __init__(self, spec: TaskSpec, env_index: int = 0, robot: str = "robot", fall_height_m: float = 0.4) -> None:
        self.spec, self.i, self.robot, self.fall_height = spec, env_index, robot, fall_height_m
        self.builder = WorldBuilder(spec)

    def reset(self) -> None:
        self.builder.reset()

    def _scene(self, env: Any) -> Any:
        return getattr(getattr(env, "unwrapped", env), "scene", env)

    def read(self, env: Any, t: float, said: list[str] | None = None, judge: dict[str, bool] | None = None) -> dict[str, Any]:
        scene = self._scene(env)
        poses: dict[str, Pose] = {}
        joints: dict[str, float] = {}
        for name, obj in self.spec.objects.items():
            ent = scene[name]
            poses[name] = Pose.of(_np(ent.data.root_pos_w)[self.i], _np(ent.data.root_quat_w)[self.i])
            if obj.get("joint"):
                idx = list(ent.joint_names).index(obj["joint"])
                joints[name] = float(_np(ent.data.joint_pos)[self.i][idx])
        base = None
        try:
            base = float(_np(scene[self.robot].data.root_pos_w)[self.i][2])
        except (KeyError, AttributeError, IndexError):
            pass
        return self.builder.build(t, poses, joints, base, self.fall_height, said, judge=judge)


class RealRoomAdapter:
    """World state from real-room measurements, one frame at a time.

    A measurement frame::

        {
          "t": 12.34,
          "tags": {"7": {"pos": [...], "quat": [w, x, y, z]}},     # camera-to-world solved poses of AprilTags
          "mocap": {"mug": {"pos": [...], "quat": [...]}},          # optional; overrides tags for that object
          "encoders": {"drawer_encoder": 0.004},                    # metres or radians
          "scales": {"drawer_scale": 1.85},                         # kilograms, gross
          "robot": {"base_height_m": 0.78},
          "speech": ["Which cup, the red or the blue one?"],
          "judge": {"Is the cloth folded in half?": true},          # answers from the AI judge, if used
        }
    """

    def __init__(self, spec: TaskSpec, cell: dict[str, Any] | str | Path, fall_height_m: float = 0.4) -> None:
        self.spec = spec
        self.cell = yaml.safe_load(Path(cell).read_text(encoding="utf-8")) if isinstance(cell, (str, Path)) else cell
        self.fall_height = fall_height_m
        self.builder = WorldBuilder(spec)
        self.fixtures = {k: Pose.of(v["pos"], v.get("quat") or quat_from_yaw(float(v.get("yaw_deg", 0.0))))
                         for k, v in (self.cell.get("fixtures") or {}).items()}
        self.said: list[str] = []
        self._last: dict[str, Pose] = {}

    def reset(self) -> None:
        self.builder.reset()
        self.said, self._last = [], {}

    def _from_tag(self, name: str, tags: dict[str, Any]) -> Pose | None:
        spec = (self.cell.get("tags") or {}).get(name)
        if not spec:
            return None
        det = tags.get(str(spec["id"]))
        if det is None:
            return None
        tag = Pose.of(det["pos"], det.get("quat"))
        offset = spec.get("offset", [0.0, 0.0, 0.0])  # object centre, in the tag's frame
        return Pose(from_frame(offset, tag), tag.quat)

    def update(self, frame: dict[str, Any]) -> dict[str, Any]:
        poses: dict[str, Pose] = dict(self.fixtures)
        tags, mocap = frame.get("tags") or {}, frame.get("mocap") or {}
        for name in self.spec.objects:
            if name in mocap:
                m = mocap[name]
                poses[name] = Pose.of(m["pos"], m.get("quat"))
            else:
                p = self._from_tag(name, tags)
                if p is not None:
                    poses[name] = p
                elif name in self._last:  # occluded this frame: keep the last seen pose
                    poses[name] = self._last[name]
        self._last = {k: v for k, v in poses.items() if k not in self.fixtures}
        joints: dict[str, float] = {}
        for name, obj in self.spec.objects.items():
            enc = obj.get("real_joint")
            if enc and enc in (frame.get("encoders") or {}):
                joints[name] = float(frame["encoders"][enc])
        extra: dict[str, list[str]] = {}
        for scale, cfg in (self.cell.get("scales") or {}).items():
            kg = (frame.get("scales") or {}).get(scale)
            if kg is not None:
                extra.setdefault(cfg["region"], []).extend(
                    on_scale(float(kg) - float(cfg.get("tare_kg", 0.0)), cfg["objects"], float(cfg.get("tol_kg", 0.02))))
        self.said += list(frame.get("speech") or [])
        base = (frame.get("robot") or {}).get("base_height_m")
        return self.builder.build(float(frame["t"]), poses, joints, base, self.fall_height, self.said,
                                  extra_regions=extra, judge=frame.get("judge"))


def on_scale(net_kg: float, masses: dict[str, float], tol_kg: float) -> list[str]:
    """Objects that must be on a scale reading ``net_kg``: in every subset of ``masses`` that sums to it.

    Ambiguous readings (two subsets weigh the same) only claim the objects common to all of them, so a scale
    never says an object is in a region unless the weight proves it.
    """
    names = sorted(masses)
    fits = [{n for i, n in enumerate(names) if mask >> i & 1} for mask in range(1, 1 << len(names))
            if abs(sum(masses[n] for i, n in enumerate(names) if mask >> i & 1) - net_kg) <= tol_kg]
    return sorted(set.intersection(*fits)) if fits else []


def tag_pose_for(obj_world: Pose, offset: list[float]) -> dict[str, Any]:
    """Where a tag mounted at ``offset`` (object frame) appears in the world: for building test frames."""
    pos = obj_world.pos - quat_to_matrix(obj_world.quat) @ np.asarray(offset, dtype=float)
    return {"pos": pos.tolist(), "quat": obj_world.quat.tolist()}
