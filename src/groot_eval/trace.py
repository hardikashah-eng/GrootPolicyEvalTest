"""Step-level traces: one data format for simulation rollouts and real-robot runs.

Comparing sim and real below the level of "did it succeed" needs both sides
logged the same way. A trace is a **LeRobot v2 dataset** (the format GR00T
fine-tuning data already uses, so ``open-loop`` and other LeRobot tools read
it unchanged) with a few extra per-step columns and one extra metadata file:

=========================  ====================================================
Column                     Meaning
=========================  ====================================================
``observation.state``      every ``state.*`` observation key, concatenated in
                           the order given by ``meta/modality.json``
``action``                 the action the policy returned for this step
``timestamp``              control time in seconds (``step * control_dt`` in
                           sim; the robot clock for real runs)
``wall_time``              seconds since the episode started, by the host clock
``inference_latency_ms``   time the policy call took (NaN if unknown)
``chunk_index``            which predicted action chunk this action came from
``chunk_step``             position inside that chunk (0 = fresh inference);
                           both -1 for policies without chunking
``reward``                 env reward (NaN if unknown)
``image.<key>.<stat>``     per-frame image statistics for each ``video.<key>``:
                           ``brightness``, ``contrast``, ``sharpness``
                           (Laplacian variance, lower = blurrier), ``height``,
                           ``width``, ``max_value`` (raw pixel max: tells a
                           0-1 float image from a 0-255 one)
=========================  ====================================================

``meta/trace_info.json`` records ``domain`` (``sim`` or ``real``), env,
policy, control period and the observation keys. ``meta/episodes.jsonl`` lines
also carry ``success``, ``failure_reason`` and the scenario when known.

Use :class:`TraceRecorder` inside any control loop (the closed-loop runner
does this for ``--trace``; a real-robot loop calls the same three methods), or
:func:`import_real` to convert an existing LeRobot recording.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from groot_eval.open_loop import LeRobotDataset, write_lerobot_dataset
from groot_eval.types import Finding

SCHEMA_VERSION = 1
TRACE_INFO = "trace_info.json"
STEP_COLUMNS = ("wall_time", "inference_latency_ms", "chunk_index", "chunk_step", "reward")
IMAGE_STATS = ("brightness", "contrast", "sharpness", "height", "width", "max_value")
_LEROBOT_COLUMNS = {"observation.state", "action", "frame_index", "episode_index", "index", "task_index"}


def image_stats(frame: Any) -> dict[str, float]:
    """Cheap per-frame statistics that expose camera and rendering differences.

    Brightness and contrast are on a 0-1 scale whatever the input dtype, so a
    float image in [0, 1] and the same image in uint8 give the same numbers;
    ``max_value`` keeps the raw range so the dtype mismatch itself is visible.
    """
    arr = np.asarray(frame)
    if arr.ndim == 4:  # (1, H, W, C) batch of one
        arr = arr[0]
    raw_max = float(np.max(arr)) if arr.size else math.nan
    img = arr.astype(np.float64)
    if np.issubdtype(arr.dtype, np.floating) and raw_max <= 1.0:
        img = img * 255.0
    gray = img.mean(axis=-1) if img.ndim == 3 else img
    if gray.shape[0] >= 3 and gray.shape[1] >= 3:
        lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]) / 255.0
        sharpness = float(lap.var())
    else:
        sharpness = math.nan
    return {
        "brightness": float(gray.mean() / 255.0),
        "contrast": float(gray.std() / 255.0),
        "sharpness": sharpness,
        "height": float(gray.shape[0]),
        "width": float(gray.shape[1]) if gray.ndim > 1 else math.nan,
        "max_value": raw_max,
    }


@dataclass
class _Episode:
    task: str
    meta: dict[str, Any]
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    columns: dict[str, list[float]] = field(default_factory=dict)
    t0: float | None = None


class TraceRecorder:
    """Collect steps from any control loop and write them as a trace.

    Typical real-robot loop::

        rec = TraceRecorder(domain="real", env="ur5_cell", policy="groot@robot", control_dt=1 / 30)
        rec.start_episode("pick up the cup")
        while running:
            obs = robot.observe()
            t0 = time.perf_counter(); action = policy.act(obs); ms = (time.perf_counter() - t0) * 1000
            rec.record_step(obs, action, latency_ms=ms, timestamp=robot.clock())
            robot.apply(action)
        rec.end_episode(success=operator_says_success)
        rec.write("traces/real")
    """

    def __init__(
        self,
        domain: str = "sim",
        env: str = "",
        policy: str = "",
        control_dt: float | None = None,
        state_keys: list[str] | None = None,
        action_groups: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        if domain not in ("sim", "real"):
            raise ValueError(f"domain must be sim or real, got {domain!r}")
        self.domain, self.env, self.policy = domain, env, policy
        self.control_dt = control_dt
        self.state_keys = list(state_keys) if state_keys else None
        self.state_dims: dict[str, int] = {}
        self.video_keys: list[str] | None = None
        self.action_groups = action_groups
        self.action_dim: int | None = None
        self.episodes: list[_Episode] = []
        self._current: _Episode | None = None

    # -- recording ------------------------------------------------------------
    def start_episode(self, task: str = "", **meta: Any) -> None:
        if self._current is not None:
            self.end_episode()
        self._current = _Episode(task=task, meta=dict(meta))

    def record_step(
        self,
        obs: dict[str, Any],
        action: Any,
        *,
        latency_ms: float = math.nan,
        reward: float = math.nan,
        wall_time: float | None = None,
        timestamp: float | None = None,
        chunk_index: int = -1,
        chunk_step: int = -1,
        extra: dict[str, float] | None = None,
    ) -> None:
        """Record the observation the policy saw and the action it returned.

        ``extra`` adds scalar columns for this step (for example ``fault.active``
        from :class:`~groot_eval.degradation.faults.FaultEnv`).

        ``wall_time`` is any monotonic clock reading (``time.perf_counter()``);
        it is stored relative to the episode's first step.
        """
        ep = self._current
        if ep is None:
            raise RuntimeError("call start_episode() before record_step()")
        if self.state_keys is None:
            self.state_keys = sorted(k for k in obs if k.startswith("state."))
        if self.video_keys is None:
            self.video_keys = sorted(k for k in obs if k.startswith("video."))
        parts = []
        for key in self.state_keys:
            if key not in obs:
                raise KeyError(f"observation is missing {key!r} (recorded earlier in this trace)")
            v = np.asarray(obs[key], dtype=np.float32).reshape(-1)
            if self.state_dims.setdefault(key, v.size) != v.size:
                raise ValueError(f"{key} changed size from {self.state_dims[key]} to {v.size}")
            parts.append(v)
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.action_dim is None:
            self.action_dim = act.size
        elif act.size != self.action_dim:
            raise ValueError(f"action changed size from {self.action_dim} to {act.size}")

        now = time.perf_counter() if wall_time is None else float(wall_time)
        if ep.t0 is None:
            ep.t0 = now
        step = len(ep.states)
        if timestamp is None:
            timestamp = step * self.control_dt if self.control_dt else now - ep.t0
        ep.states.append(np.concatenate(parts) if parts else np.zeros(0, np.float32))
        ep.actions.append(act)
        row = {
            "timestamp": float(timestamp),
            "wall_time": now - ep.t0,
            "inference_latency_ms": float(latency_ms),
            "chunk_index": float(chunk_index),
            "chunk_step": float(chunk_step),
            "reward": float(reward),
        }
        for k, v in (extra or {}).items():
            row[str(k)] = float(v)
        for key in self.video_keys:
            if key in obs:
                for stat, value in image_stats(obs[key]).items():
                    row[f"image.{key.removeprefix('video.')}.{stat}"] = value
        for k, v in row.items():
            ep.columns.setdefault(k, [math.nan] * step).append(v)
        for k, values in ep.columns.items():  # a column missing on this step (e.g. a dropped frame)
            if len(values) < step + 1:
                values.append(math.nan)

    def end_episode(self, success: bool | None = None, failure_reason: str | None = None, **meta: Any) -> None:
        ep = self._current
        if ep is None:
            return
        ep.meta.update(meta)
        if success is not None:
            ep.meta["success"] = bool(success)
        if failure_reason is not None:
            ep.meta["failure_reason"] = str(failure_reason)
        if ep.states:
            self.episodes.append(ep)
        self._current = None

    # -- output ---------------------------------------------------------------
    def modality(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        start = 0
        for key in self.state_keys or []:
            n = self.state_dims[key]
            state[key.removeprefix("state.")] = {"start": start, "end": start + n}
            start += n
        groups = self.action_groups or {"action": (0, self.action_dim or 0)}
        mod: dict[str, Any] = {
            "state": state,
            "action": {k: {"start": int(a), "end": int(b)} for k, (a, b) in groups.items()},
            "annotation": {"task": {}},
        }
        if self.video_keys:
            mod["video"] = {k.removeprefix("video."): {} for k in self.video_keys}
        return mod

    def write(self, root: str | Path) -> Path:
        if self._current is not None:
            self.end_episode()
        if not self.episodes:
            raise ValueError("no episodes recorded")
        fps = 1.0 / self.control_dt if self.control_dt else _estimate_fps(self.episodes)
        payload = [
            {"task": ep.task, "states": np.stack(ep.states), "actions": np.stack(ep.actions),
             "extra": {k: np.asarray(v, dtype=np.float64) for k, v in ep.columns.items()}, "meta": _plain(ep.meta)}
            for ep in self.episodes
        ]
        root = write_lerobot_dataset(root, payload, fps=_fps_value(fps), modality=self.modality())
        info = {
            "schema_version": SCHEMA_VERSION,
            "domain": self.domain,
            "env": self.env,
            "policy": self.policy,
            "control_dt": self.control_dt,
            "fps": fps,
            "state_keys": self.state_keys,
            "video_keys": self.video_keys or [],
            "action_dim": self.action_dim,
            "episodes": len(self.episodes),
        }
        (Path(root) / "meta" / TRACE_INFO).write_text(json.dumps(info, indent=2), encoding="utf-8")
        return Path(root)


def _estimate_fps(episodes: list[_Episode]) -> float:
    diffs = np.concatenate([np.diff(e.columns["timestamp"]) for e in episodes if len(e.states) > 1] or [np.array([])])
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    return float(1.0 / np.median(diffs)) if diffs.size else 1.0


def _fps_value(fps: float) -> float | int:
    return int(round(fps)) if abs(fps - round(fps)) < 1e-6 else float(fps)


def _plain(value: Any) -> Any:
    """JSON-safe copy (numpy scalars to Python, NaN to None)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# -- reading ------------------------------------------------------------------
@dataclass
class TraceSet:
    """A trace (or any LeRobot v2 dataset) opened for comparison."""

    dataset: LeRobotDataset
    info: dict[str, Any]
    _frames: list[pd.DataFrame] | None = None

    @classmethod
    def open(cls, root: str | Path) -> "TraceSet":
        ds = LeRobotDataset.open(root)
        path = Path(root) / "meta" / TRACE_INFO
        info = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return cls(ds, info)

    @property
    def root(self) -> Path:
        return self.dataset.root

    @property
    def domain(self) -> str:
        return self.info.get("domain", "unknown")

    @property
    def fps(self) -> float | None:
        fps = self.info.get("fps") or self.dataset.info.get("fps")
        return float(fps) if fps else None

    @property
    def is_trace(self) -> bool:
        return bool(self.info)

    def frames(self) -> list[pd.DataFrame]:
        if self._frames is None:
            self._frames = [self.dataset.episode_frame(e["episode_index"]) for e in self.dataset.episodes]
        return self._frames

    def stacked(self, column: str) -> np.ndarray:
        """All steps of a vector column (``observation.state``/``action``) as (N, D)."""
        rows = [np.stack(f[column].to_numpy()) for f in self.frames() if column in f and len(f)]
        return np.concatenate(rows).astype(np.float64) if rows else np.zeros((0, 0))

    def per_episode(self, column: str) -> list[np.ndarray]:
        return [np.stack(f[column].to_numpy()).astype(np.float64) for f in self.frames() if column in f and len(f)]

    def state_slices(self) -> dict[str, slice]:
        mod = self.dataset.modality.get("state")
        if mod:
            return {k: slice(v["start"], v["end"]) for k, v in sorted(mod.items(), key=lambda kv: kv[1]["start"])}
        dim = self.stacked("observation.state").shape[1] if self.frames() else 0
        return {"state": slice(0, dim)}

    def action_slices(self) -> dict[str, slice]:
        dim = self.stacked("action").shape[1] if self.frames() else 0
        mod = self.dataset.modality.get("action")
        if mod:
            return {k: slice(v["start"], v["end"]) for k, v in sorted(mod.items(), key=lambda kv: kv[1]["start"])}
        return {"action": slice(0, dim)}

    def scalar_columns(self, prefix: str) -> list[str]:
        cols: set[str] = set()
        for f in self.frames():
            cols.update(c for c in f.columns if c.startswith(prefix))
        return sorted(cols)

    def scalar(self, column: str) -> np.ndarray:
        parts = [f[column].to_numpy(dtype=float) for f in self.frames() if column in f]
        return np.concatenate(parts) if parts else np.zeros(0)

    def episode_meta(self) -> list[dict[str, Any]]:
        return list(self.dataset.episodes)


def dim_names(slices: dict[str, slice]) -> list[str]:
    """Human-readable names per vector dimension, e.g. ``left_arm[3]``."""
    names = []
    for key, sl in slices.items():
        width = sl.stop - sl.start
        names += [key] if width == 1 else [f"{key}[{i}]" for i in range(width)]
    return names


# -- importing real recordings ------------------------------------------------
def import_real(src: str | Path, out: str | Path, video: bool = False, domain: str = "real",
                env: str = "", policy: str = "") -> tuple[Path, list[Finding]]:
    """Convert a LeRobot v2 recording into a trace and report what is missing or suspicious.

    Columns the recording lacks are added (latency and chunk columns as
    NaN / -1; ``wall_time`` from ``timestamp``). With ``video=True`` the camera
    streams in ``modality.json`` are decoded once to compute the
    ``image.*`` statistics.
    """
    ds = LeRobotDataset.open(src)
    findings: list[Finding] = []
    fps = float(ds.info.get("fps") or 0) or None
    if fps is None:
        findings.append(Finding("error", "fps", "meta/info.json has no fps; timing checks are impossible"))
    video_keys = list((ds.modality.get("video") or {}).keys()) if video else []
    if video and not video_keys:
        findings.append(Finding("warn", "video", "--video given but modality.json lists no video keys"))
    payload = []
    jitter, dt_ratio = [], []
    for meta in ds.episodes:
        idx = meta["episode_index"]
        try:
            frame = ds.episode_frame(idx)
        except FileNotFoundError:
            findings.append(Finding("error", "files", f"episode {idx}: parquet file missing"))
            continue
        missing = {"observation.state", "action"} - set(frame.columns)
        if missing:
            findings.append(Finding("error", "columns", f"episode {idx}: missing {sorted(missing)}"))
            continue
        n = len(frame)
        if meta.get("length") is not None and int(meta["length"]) != n:
            findings.append(Finding("warn", "length", f"episode {idx}: episodes.jsonl says {meta['length']} steps, parquet has {n}"))
        states = np.stack(frame["observation.state"].to_numpy()).astype(np.float32)
        actions = np.stack(frame["action"].to_numpy()).astype(np.float32)
        for name, arr in (("observation.state", states), ("action", actions)):
            bad = int((~np.isfinite(arr)).any(axis=1).sum())
            if bad:
                findings.append(Finding("error", "finite", f"episode {idx}: {bad} steps with NaN/inf in {name}"))
        if "timestamp" in frame:
            ts = frame["timestamp"].to_numpy(dtype=float)
        else:
            ts = np.arange(n) / (fps or 1.0)
            findings.append(Finding("warn", "timestamp", f"episode {idx}: no timestamp column; derived from fps"))
        d = np.diff(ts)
        if d.size and np.any(d <= 0):
            findings.append(Finding("error", "timestamp", f"episode {idx}: timestamps not strictly increasing ({int(np.sum(d <= 0))} steps)"))
        if d.size and fps:
            dt_ratio.append(float(np.median(d) * fps))
            jitter.append(float(np.std(d) * 1000))
        extra: dict[str, np.ndarray] = {"timestamp": ts}
        extra["wall_time"] = frame["wall_time"].to_numpy(dtype=float) if "wall_time" in frame else ts - (ts[0] if n else 0)
        for col, default in (("inference_latency_ms", math.nan), ("chunk_index", -1.0), ("chunk_step", -1.0), ("reward", math.nan)):
            extra[col] = frame[col].to_numpy(dtype=float) if col in frame else np.full(n, default)
        for col in frame.columns:  # keep every other per-step scalar (image stats, fault.*, camera_time.*, ...)
            if col in extra or col in _LEROBOT_COLUMNS:
                continue
            try:
                extra[col] = frame[col].to_numpy(dtype=float)
            except (TypeError, ValueError):
                findings.append(Finding("warn", "columns", f"episode {idx}: column {col!r} is not a numeric scalar; not imported"))
        for key in video_keys:
            extra.update(_video_stats(ds, idx, key, n, findings))
        keep = {k: v for k, v in meta.items() if k not in ("episode_index", "tasks", "length")}
        payload.append({"task": (meta.get("tasks") or [""])[0], "states": states, "actions": actions,
                        "extra": extra, "meta": _plain(keep)})
    if not payload:
        raise ValueError(f"no usable episodes in {src}")
    if dt_ratio:
        worst = max(dt_ratio, key=lambda r: abs(r - 1))
        if abs(worst - 1) > 0.05:
            findings.append(Finding("error", "fps", f"median timestamp step is {worst:.2f}x the 1/fps={1 / fps:.4f}s period in at least one episode: "
                                                    "the declared control rate does not match the recording"))
        findings.append(Finding("ok" if max(jitter) < 0.2 * 1000 / fps else "warn", "jitter",
                                f"timestamp jitter (std of step period) up to {max(jitter):.2f} ms at {fps:g} Hz"))
    root = write_lerobot_dataset(out, payload, fps=_fps_value(fps or 1.0), modality=ds.modality or None)
    state_keys = [f"state.{k}" for k in (ds.modality.get("state") or {"state": {}})]
    src_info_path = Path(src) / "meta" / TRACE_INFO
    src_info = json.loads(src_info_path.read_text(encoding="utf-8")) if src_info_path.exists() else {}
    env, policy = env or src_info.get("env", ""), policy or src_info.get("policy", "")
    info = {**src_info,  # provenance (robot_id, calibration_id, robot_age_hours, action_source, ...) survives re-import
            "schema_version": SCHEMA_VERSION, "domain": domain, "env": env, "policy": policy,
            "control_dt": 1.0 / fps if fps else None, "fps": fps, "state_keys": state_keys,
            "video_keys": [f"video.{k}" for k in video_keys], "action_dim": int(payload[0]["actions"].shape[1]),
            "episodes": len(payload), "imported_from": str(src)}
    (Path(root) / "meta" / TRACE_INFO).write_text(json.dumps(info, indent=2), encoding="utf-8")
    findings.append(Finding("ok", "import", f"{len(payload)} episodes written to {root}"))
    return Path(root), findings


def _video_stats(ds: LeRobotDataset, idx: int, key: str, n: int, findings: list[Finding]) -> dict[str, np.ndarray]:
    from groot_eval.open_loop import _VideoReader

    cols = {f"image.{key}.{s}": np.full(n, math.nan) for s in IMAGE_STATS}
    try:
        reader = _VideoReader(ds.video_path(idx, key))
    except FileNotFoundError:
        findings.append(Finding("error", "video", f"episode {idx}: video for {key} not found at {ds.video_path(idx, key)}"))
        return cols
    try:
        count = 0
        for i, frame in enumerate(reader.iter_frames()):
            if i >= n:
                count = i + 1
                continue
            for stat, value in image_stats(frame).items():
                cols[f"image.{key}.{stat}"][i] = value
            count = i + 1
    finally:
        reader.close()
    if count != n:
        findings.append(Finding("warn", "video", f"episode {idx}: {key} has {count} frames for {n} steps (camera rate or dropped frames)"))
    return cols
