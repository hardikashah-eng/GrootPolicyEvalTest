"""Build the scorer's world state from poses and readings. Shared by the Isaac Lab and real-room adapters.

Both adapters reduce what they measure to the same things: a pose per object or fixture, joint readings, the
robot's base height, and what the robot said. ``WorldBuilder`` turns those into the world state that
``spec.evaluate`` reads, using the region boxes, targets and route declared in the task spec. Because the
geometry lives in the spec and this code is shared, sim and real are scored by the same rules.

Poses are (position xyz in metres, quaternion wxyz), in the world frame. Isaac Lab uses wxyz too.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from livechorebench.spec import TaskSpec


@dataclass
class Pose:
    pos: np.ndarray
    quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    @staticmethod
    def of(pos: Any, quat: Any = None) -> Pose:
        q = np.array([1.0, 0.0, 0.0, 0.0]) if quat is None else np.asarray(quat, dtype=float)
        return Pose(np.asarray(pos, dtype=float).reshape(3), q / (np.linalg.norm(q) or 1.0))


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_yaw(yaw_deg: float) -> np.ndarray:
    h = math.radians(yaw_deg) / 2
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])


def tilt_deg(q: np.ndarray) -> float:
    """Angle between the body's up axis and world up."""
    up = quat_to_matrix(q)[:, 2]
    return math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))


def yaw_deg(q: np.ndarray) -> float:
    R = quat_to_matrix(q)
    return math.degrees(math.atan2(R[1, 0], R[0, 0]))


def to_frame(point: np.ndarray, frame: Pose) -> np.ndarray:
    """World point -> coordinates in ``frame``."""
    return quat_to_matrix(frame.quat).T @ (np.asarray(point, dtype=float) - frame.pos)


def from_frame(point: Any, frame: Pose) -> np.ndarray:
    return quat_to_matrix(frame.quat) @ np.asarray(point, dtype=float) + frame.pos


class WorldBuilder:
    """Turn poses and readings into a world state for one task spec."""

    def __init__(self, spec: TaskSpec, still_speed_mps: float = 0.01) -> None:
        self.spec = spec
        self.regions: dict[str, Any] = dict(spec.scene.get("regions", {}))
        self.targets: dict[str, Any] = dict(spec.scene.get("targets", {}))
        self.route: dict[str, Any] | None = spec.scene.get("route")
        self.drop_below_m = float(spec.scene.get("drop_below_m", 0.2))
        self.still_speed = still_speed_mps
        self.start: dict[str, Pose] = {}
        self._last: dict[str, Pose] = {}
        self._last_t: float | None = None
        self.stable_s = 0.0
        self.dropped: set[str] = set()

    def reset(self) -> None:
        self.start, self._last, self._last_t, self.stable_s, self.dropped = {}, {}, None, 0.0, set()

    def _frame(self, name: str, poses: dict[str, Pose]) -> Pose | None:
        return poses.get(name)

    def build(self, t: float, poses: dict[str, Pose], joints: dict[str, float] | None = None,
              base_height_m: float | None = None, fall_height_m: float = 0.4, said: list[str] | None = None,
              extra_regions: dict[str, list[str]] | None = None, judge: dict[str, bool] | None = None) -> dict[str, Any]:
        """World state at time ``t``. Call once per control tick (or at least a few times a second)."""
        for k, v in poses.items():  # start pose: where each object was first seen (tags may appear late)
            if k not in self.start:
                self.start[k] = Pose(v.pos.copy(), v.quat.copy())
        # stillness: every movable object slower than still_speed since the last call
        moving = False
        if self._last_t is not None and t > self._last_t:
            dt = t - self._last_t
            for name, p in poses.items():
                prev = self._last.get(name)
                obj = self.spec.objects.get(name, {})
                if prev is not None and not obj.get("fixed") and np.linalg.norm(p.pos - prev.pos) / dt > self.still_speed:
                    moving = True
            self.stable_s = 0.0 if moving else self.stable_s + dt
        self._last = {k: Pose(v.pos.copy(), v.quat.copy()) for k, v in poses.items()}
        self._last_t = t

        objects: dict[str, Any] = {}
        for name, p in poses.items():
            objects[name] = {"pos": p.pos.tolist(), "yaw_deg": yaw_deg(p.quat), "tilt_deg": tilt_deg(p.quat)}
            obj = self.spec.objects.get(name, {})
            start = self.start.get(name)
            # dropped: fell below drop_below_m having started above it (an object that starts on the floor,
            # like a laundry basket, is not flagged by height)
            if not obj.get("fixed") and start is not None and start.pos[2] >= self.drop_below_m > p.pos[2]:
                self.dropped.add(name)

        regions: dict[str, list[str]] = {}
        for rname, r in self.regions.items():
            frame = self._frame(r["frame"], poses)
            inside = []
            if frame is not None:
                lo, hi = np.asarray(r["min"], float), np.asarray(r["max"], float)
                for name, p in poses.items():
                    if name == r["frame"] or self.spec.objects.get(name, {}).get("fixed"):
                        continue
                    local = to_frame(p.pos, frame)
                    if np.all(local >= lo) and np.all(local <= hi):
                        inside.append(name)
            regions[rname] = inside
        for rname, names in (extra_regions or {}).items():  # e.g. a scale says the item is in the drawer
            regions[rname] = sorted(set(regions.get(rname, [])) | set(names))

        targets: dict[str, Any] = {}
        for tname, tg in self.targets.items():
            frame = self._frame(tg["frame"], poses)
            if frame is not None:
                targets[tname] = {"pos": from_frame(tg["pos"], frame).tolist(),
                                  "yaw_deg": yaw_deg(frame.quat) + float(tg.get("yaw_deg", 0.0))}

        derived = dict(joints or {})
        for name, p in poses.items():
            if name in self.start:
                derived[f"{name}.height_m"] = float(p.pos[2] - self.start[name].pos[2])
        if self.route and self.route.get("carrier") in poses:
            derived["route.progress"] = route_progress(poses[self.route["carrier"]].pos, self.route["points"])

        return {"t": t, "objects": objects, "regions": regions, "targets": targets, "joints": derived,
                "stable_s": self.stable_s, "fallen": base_height_m is not None and base_height_m < fall_height_m,
                "dropped": sorted(self.dropped), "said": list(said or []), "judge": dict(judge or {})}


def route_progress(pos: np.ndarray, points: list[list[float]]) -> float:
    """Share of a 2-D polyline route covered by the point's projection (0 at the start, 1 at the end)."""
    P = np.asarray(points, dtype=float)
    seg = np.diff(P, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    total = float(lengths.sum()) or 1.0
    best_d, best_s, run = math.inf, 0.0, 0.0
    for a, d, length in zip(P[:-1], seg, lengths):
        u = 0.0 if length == 0 else float(np.clip(np.dot(np.asarray(pos[:2]) - a, d) / (length * length), 0, 1))
        q = a + u * d
        dist = float(np.linalg.norm(np.asarray(pos[:2]) - q))
        if dist < best_d:
            best_d, best_s = dist, run + u * length
        run += length
    return best_s / total
