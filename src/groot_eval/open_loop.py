"""Open-loop evaluation against recorded demonstrations (LeRobot v2 format).

The policy sees the recorded observation at step ``t``, predicts an action
chunk, and that chunk is compared with the actions the demonstrator actually
took over the next ``horizon`` steps. It is cheap (no simulator) and catches
broken checkpoints, normalization bugs and embodiment mix-ups early, but it
cannot show compounding error, so it complements closed-loop evaluation
rather than replacing it.

Dataset layout read here (the layout GR00T fine-tuning uses)::

    <root>/meta/info.json            fps, data_path template, features
    <root>/meta/episodes.jsonl       {"episode_index", "tasks", "length"}
    <root>/meta/modality.json        state/action/video/annotation slices
    <root>/data/chunk-000/episode_000000.parquet
    <root>/videos/chunk-000/<video_key>/episode_000000.mp4   (optional)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from groot_eval.policies.base import Policy
from groot_eval.stats import bootstrap_ci

DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"


@dataclass
class LeRobotDataset:
    root: Path
    info: dict[str, Any]
    episodes: list[dict[str, Any]]
    modality: dict[str, Any]

    @classmethod
    def open(cls, root: str | Path) -> "LeRobotDataset":
        root = Path(root)
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        episodes = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        modality_path = root / "meta" / "modality.json"
        modality = json.loads(modality_path.read_text(encoding="utf-8")) if modality_path.exists() else {}
        return cls(root, info, episodes, modality)

    def episode_frame(self, episode_index: int) -> pd.DataFrame:
        template = self.info.get("data_path", DEFAULT_DATA_PATH)
        chunk = episode_index // int(self.info.get("chunks_size", 1000))
        return pd.read_parquet(self.root / template.format(episode_chunk=chunk, episode_index=episode_index))

    def video_path(self, episode_index: int, video_key: str) -> Path:
        template = self.info.get("video_path", DEFAULT_VIDEO_PATH)
        chunk = episode_index // int(self.info.get("chunks_size", 1000))
        return self.root / template.format(episode_chunk=chunk, video_key=video_key, episode_index=episode_index)

    # -- observation assembly ----------------------------------------------
    def observation(self, frame: pd.DataFrame, t: int, task: str, video: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
        row = frame.iloc[t]
        state = np.asarray(row["observation.state"], dtype=np.float32)
        obs: dict[str, Any] = {}
        state_mod = self.modality.get("state") or {"state": {"start": 0, "end": len(state)}}
        for key, sl in state_mod.items():
            obs[f"state.{key}"] = state[sl["start"] : sl["end"]]
        for key in (self.modality.get("annotation") or {"task": {}}):
            obs[f"annotation.{key}"] = task
        if video:
            obs.update({f"video.{k}": v for k, v in video.items()})
        return obs

    def action_slices(self, action_dim: int) -> dict[str, slice]:
        mod = self.modality.get("action")
        if not mod:
            return {"action": slice(0, action_dim)}
        return {key: slice(sl["start"], sl["end"]) for key, sl in mod.items()}


class _VideoReader:
    """Sequential frame access via OpenCV, only when a dataset has videos and a policy needs them."""

    def __init__(self, path: Path) -> None:
        import cv2  # optional dependency: pip install groot-policy-eval[video]

        self._cv2 = cv2
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(path)

    def frame(self, index: int) -> np.ndarray:
        self.cap.set(self._cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = self.cap.read()
        if not ok:
            raise IndexError(f"frame {index} unreadable")
        return self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()


@dataclass
class OpenLoopResult:
    policy: str
    horizon: int
    per_episode: list[dict[str, Any]] = field(default_factory=list)
    per_horizon_mse: list[float] = field(default_factory=list)
    per_group_mse: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        mses = [e["mse"] for e in self.per_episode]
        lo, hi = bootstrap_ci(mses) if mses else (float("nan"), float("nan"))
        return {
            "policy": self.policy,
            "horizon": self.horizon,
            "episodes": len(self.per_episode),
            "mse_mean": float(np.mean(mses)) if mses else float("nan"),
            "mse_ci_low": lo,
            "mse_ci_high": hi,
            "mae_mean": float(np.mean([e["mae"] for e in self.per_episode])) if mses else float("nan"),
            "per_group_mse": self.per_group_mse,
            "per_horizon_mse": self.per_horizon_mse,
            "worst_episodes": sorted(self.per_episode, key=lambda e: -e["mse"])[:5],
        }


def evaluate_open_loop(
    dataset: LeRobotDataset,
    policy: Policy | Any,
    horizon: int = 16,
    episodes: list[int] | None = None,
    use_video: bool = False,
) -> OpenLoopResult:
    """Score predicted action chunks against recorded actions.

    ``policy`` may be a chunk predictor (``predict_chunk``) or a per-step
    policy (``act``); a per-step policy is fed the recorded observation for
    each step of the window, which is the usual teacher-forced setup.
    """
    chunked = hasattr(policy, "predict_chunk")
    name = getattr(policy, "name", type(policy).__name__)
    result = OpenLoopResult(policy=name, horizon=horizon)
    horizon_sq: list[list[float]] = [[] for _ in range(horizon)]
    group_sq: dict[str, list[float]] = {}  # [sum of squared error, element count]
    video_keys = list((dataset.modality.get("video") or {}).keys()) if use_video else []

    ep_meta = {e["episode_index"]: e for e in dataset.episodes}
    for ep in episodes if episodes is not None else sorted(ep_meta):
        frame = dataset.episode_frame(ep)
        task = (ep_meta.get(ep, {}).get("tasks") or [""])[0]
        gt = np.stack(frame["action"].to_numpy()).astype(float)
        readers = {k: _VideoReader(dataset.video_path(ep, k)) for k in video_keys}
        try:
            policy.reset(task)
            errors = []
            for t0 in range(0, len(frame), horizon):
                window = gt[t0 : t0 + horizon]
                video = {k: r.frame(t0)[None] for k, r in readers.items()} or None
                if chunked:
                    pred = np.asarray(policy.predict_chunk(dataset.observation(frame, t0, task, video)), dtype=float)[: len(window)]
                else:
                    pred = np.stack([
                        np.asarray(policy.act(dataset.observation(frame, t, task, {k: r.frame(t)[None] for k, r in readers.items()} or None)), dtype=float)
                        for t in range(t0, t0 + len(window))
                    ])
                if pred.shape != window.shape:
                    raise ValueError(f"episode {ep} step {t0}: predicted {pred.shape}, recorded {window.shape}")
                err = pred - window
                errors.append(err)
                for h, row in enumerate(err):
                    horizon_sq[h].append(float(np.mean(row**2)))
                for group, sl in dataset.action_slices(gt.shape[1]).items():
                    acc = group_sq.setdefault(group, [0.0, 0.0])
                    acc[0] += float(np.sum(err[:, sl] ** 2))
                    acc[1] += err[:, sl].size
        finally:
            for r in readers.values():
                r.close()
        all_err = np.concatenate(errors)
        result.per_episode.append({
            "episode_index": ep,
            "task": task,
            "steps": len(frame),
            "mse": float(np.mean(all_err**2)),
            "mae": float(np.mean(np.abs(all_err))),
            "max_abs_error": float(np.max(np.abs(all_err))),
        })
    result.per_horizon_mse = [float(np.mean(v)) if v else float("nan") for v in horizon_sq]
    result.per_group_mse = {k: (s / n if n else float("nan")) for k, (s, n) in group_sq.items()}
    return result


def write_lerobot_dataset(
    root: str | Path,
    episodes: list[dict[str, Any]],
    fps: int = 20,
    modality: dict[str, Any] | None = None,
) -> Path:
    """Write episodes (``{"task", "states": (T,S), "actions": (T,A)}``) as a LeRobot v2 dataset."""
    root = Path(root)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    lines = []
    for i, ep in enumerate(episodes):
        states, actions = np.asarray(ep["states"], dtype=np.float32), np.asarray(ep["actions"], dtype=np.float32)
        path = root / DEFAULT_DATA_PATH.format(episode_chunk=0, episode_index=i)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "observation.state": list(states),
            "action": list(actions),
            "timestamp": np.arange(len(states), dtype=np.float32) / fps,
            "frame_index": np.arange(len(states)),
            "episode_index": i,
            "index": np.arange(len(states)),
            "task_index": 0,
        }).to_parquet(path)
        lines.append(json.dumps({"episode_index": i, "tasks": [ep.get("task", "")], "length": len(states)}))
    (root / "meta" / "episodes.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    state_dim = int(np.asarray(episodes[0]["states"]).shape[1])
    action_dim = int(np.asarray(episodes[0]["actions"]).shape[1])
    info = {
        "codebase_version": "v2.0",
        "fps": fps,
        "chunks_size": 1000,
        "total_episodes": len(episodes),
        "data_path": DEFAULT_DATA_PATH,
        "video_path": DEFAULT_VIDEO_PATH,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [state_dim]},
            "action": {"dtype": "float32", "shape": [action_dim]},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if modality:
        (root / "meta" / "modality.json").write_text(json.dumps(modality, indent=2), encoding="utf-8")
    return root
