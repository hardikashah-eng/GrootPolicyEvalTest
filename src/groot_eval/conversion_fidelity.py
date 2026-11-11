"""HDF5 -> LeRobot conversion fidelity: did the conversion lose information, visual or actuator?

Isaac Lab (Mimic, the recorder manager) and robomimic write demonstrations as
HDF5; GR00T fine-tuning reads LeRobot v2 (parquet + mp4). Every conversion is a
chance to lose something without noticing: a float64 joint stream rounded to
float32 is fine, one resampled from 60 to 30 Hz is not; an RGB camera saved as
BGR, re-encoded at low quality, resized, or shifted by one frame trains a
policy on a different world. :mod:`groot_eval.transfer` (T0) compares two
LeRobot datasets; this module compares the HDF5 *source* with its LeRobot
*output*, value by value and frame by frame.

Gates, cheapest first (same exit codes as ``fidelity``: 0 pass, 1 fail, 2 a
required gate could not run):

====  ==========  =====================================================================
C0    inventory   episodes and lengths match; every HDF5 dataset is converted or
                  deliberately ignored (anything else is reported as lost: depth,
                  force/torque, contact, extra cameras); episode attributes survive
C1    timing      control rate kept, no resampling, timestamps regular, each video
                  has one frame per row, numeric streams not shifted by a step
C2    actuator    per state/action dimension: max error relative to the signal's
                  spread, SNR (dB) and effective bits, plus the usual culprits:
                  joint order permuted, sign flipped, degrees vs radians or another
                  scale, quantization, clipping, high-frequency loss (filtering),
                  velocities (first differences) preserved, NaN introduced
C3    visual      per camera: resolution, RGB/BGR order, value range, frame count
                  and alignment, PSNR and SSIM, sharpness (detail lost to codec or
                  scaling), colour shift, frozen or duplicated frames; optionally the
                  distance in the policy's own embedding space
C4    metadata    task strings / language instructions, success labels, modality.json
                  slices and info.json shapes consistent with the data
====  ==========  =====================================================================

The mapping from HDF5 keys to LeRobot columns comes from a YAML file (see
``configs/conversion/mapping_example.yaml``) or is inferred from
``modality.json`` key names. :func:`convert_hdf5_to_lerobot` is a small
reference converter: run it next to a production converter and compare both
outputs with this module to tell pipeline bugs from unavoidable codec loss.
"""

from __future__ import annotations

import fnmatch
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from groot_eval import twin_metrics as tm
from groot_eval.fidelity import GateResult, _status, exit_code
from groot_eval.types import Finding

DEFAULTS: dict[str, dict[str, Any]] = {
    "C0_inventory": {"required": True, "lost_is_error": False},
    "C1_timing": {"required": True, "max_fps_rel_error": 0.01, "max_period_jitter_frac": 0.1},
    "C2_actuator": {"required": True, "max_rel_error": 1e-4, "min_snr_db": 60.0, "min_level_ratio": 0.5,
                    "max_pinned_increase": 0.01, "min_hf_ratio": 0.8, "hf_cut": 0.25, "max_diff_nrmse": 1e-3},
    "C3_visual": {"required": True, "min_psnr_mean": 35.0, "min_psnr_p5": 30.0, "min_ssim_mean": 0.95,
                  "min_sharpness_ratio": 0.8, "max_color_shift": 3.0, "max_frozen_frac": 0.01,
                  "sample_every": 1, "max_frames_per_episode": 300, "max_embedding_distance": 0.05},
    "C4_metadata": {"required": True},
}
WHAT = {
    "C0_inventory": "episodes, lengths and every HDF5 dataset accounted for",
    "C1_timing": "control rate, no resampling, regular timestamps, frames per row, no step shift",
    "C2_actuator": "state and action values preserved: error, SNR, bits, order, sign, units, quantization, "
                   "clipping, filtering, velocities",
    "C3_visual": "camera frames preserved: resolution, channel order, range, alignment, PSNR, SSIM, sharpness, colour",
    "C4_metadata": "task strings, success labels, modality.json and info.json consistent with the data",
}
_FORCE_LIKE = re.compile(r"(force|torque|wrench|ft_|tactile|contact|effort|current|depth|pointcloud|seg)", re.I)
_IMAGE_LIKE = re.compile(r"(rgb|image|img|cam|depth)", re.I)


def load_config(path: str | Path | None) -> dict[str, Any]:
    """Thresholds (``gates``) and the key mapping (``mapping``) from one YAML file."""
    cfg = {"gates": {k: dict(v) for k, v in DEFAULTS.items()}, "mapping": {}}
    if path:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        for k, v in (data.get("gates") or {}).items():
            if k not in cfg["gates"]:
                raise ValueError(f"unknown conversion gate {k!r}; known: {sorted(cfg['gates'])}")
            cfg["gates"][k].update(v or {})
        cfg["mapping"] = data.get("mapping") or {}
    return cfg


# =====================================================================================
# HDF5 source
# =====================================================================================
@dataclass
class H5Episode:
    name: str
    datasets: dict[str, np.ndarray]  # path relative to the episode group -> array (time on axis 0)
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        lens = [len(v) for k, v in self.datasets.items() if v.ndim >= 1 and not k.startswith("initial_state")]
        return int(np.median(lens)) if lens else 0


def _natural(name: str) -> tuple:
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name))


def _attr(v: Any) -> Any:
    if isinstance(v, bytes):
        v = v.decode("utf-8", "replace")
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, np.ndarray):
        v = v.tolist()
    if isinstance(v, str) and v[:1] in "{[":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    return v


def read_hdf5(path: str | Path, pattern: str = "data/demo_*", load_images: bool = True) -> tuple[list[H5Episode], dict[str, Any]]:
    """Read every episode group matching ``pattern`` (robomimic / Isaac Lab layout by default)."""
    try:
        import h5py
    except ImportError as exc:  # optional dependency
        raise ImportError("reading HDF5 needs h5py: pip install 'groot-policy-eval[hdf5]'") from exc
    episodes: list[H5Episode] = []
    with h5py.File(path, "r") as f:
        parent_path, _, leaf = pattern.rpartition("/")
        parent = f[parent_path] if parent_path else f
        root_attrs = {k: _attr(v) for k, v in parent.attrs.items()}
        names = sorted((n for n in parent.keys() if fnmatch.fnmatch(n, leaf)), key=_natural)
        for n in names:
            grp = parent[n]
            data: dict[str, np.ndarray] = {}

            def visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    if not load_images and obj.ndim >= 3 and obj.dtype == np.uint8:
                        data[name] = np.zeros((obj.shape[0], 0), dtype=np.uint8)  # placeholder keeps the inventory
                    else:
                        data[name] = obj[()]

            grp.visititems(visit)
            episodes.append(H5Episode(n, data, {k: _attr(v) for k, v in grp.attrs.items()}))
    return episodes, root_attrs


# =====================================================================================
# Mapping
# =====================================================================================
@dataclass
class Mapping:
    state: list[tuple[str, str]]  # (hdf5 key, lerobot modality name), concatenated in order
    action: list[tuple[str, str]]
    video: dict[str, str]  # lerobot video key -> hdf5 key
    scalars: dict[str, str]  # lerobot column -> hdf5 key
    ignore: list[str]
    episodes: str = "data/demo_*"
    fps: float | None = None
    task_attr: str | None = None
    success_attr: str | None = "success"
    inferred: list[str] = field(default_factory=list)


def _entries(spec: Any) -> list[tuple[str, str]]:
    out = []
    for item in spec or []:
        if isinstance(item, str):
            out.append((item, item.rsplit("/", 1)[-1]))
        else:
            out.append((item["key"], item.get("modality") or item["key"].rsplit("/", 1)[-1]))
    return out


def _find_key(keys: list[str], name: str) -> str | None:
    cands = [k for k in keys if k == name or k.endswith("/" + name)]
    if not cands:
        cands = [k for k in keys if k.rsplit("/", 1)[-1].lower() == name.lower()]
    return sorted(cands, key=len)[0] if cands else None


def build_mapping(spec: dict[str, Any], sample: H5Episode, modality: dict[str, Any]) -> Mapping:
    """Use the YAML mapping, filling gaps from modality.json names matched against HDF5 dataset names."""
    keys = list(sample.datasets)
    inferred: list[str] = []
    state = _entries(spec.get("state"))
    if not state and modality.get("state"):
        for name, sl in sorted(modality["state"].items(), key=lambda kv: kv[1]["start"]):
            k = _find_key(keys, name) or _find_key(keys, f"obs/{name}")
            if k:
                state.append((k, name))
                inferred.append(f"state.{name} <- {k}")
    action = _entries(spec.get("action"))
    if not action:
        if modality.get("action") and len(modality["action"]) > 1:
            for name, sl in sorted(modality["action"].items(), key=lambda kv: kv[1]["start"]):
                k = _find_key(keys, name)
                if k:
                    action.append((k, name))
        if not action:
            k = _find_key(keys, "actions") or _find_key(keys, "action")
            if k:
                action = [(k, "action")]
        inferred += [f"action.{n} <- {k}" for k, n in action]
    video = dict(spec.get("video") or {})
    if not video and modality.get("video"):
        for vk in modality["video"]:
            k = _find_key(keys, vk) or next((x for x in keys if vk.lower() in x.lower() and _IMAGE_LIKE.search(x)
                                             and sample.datasets[x].ndim == 4), None)
            if k:
                video[vk] = k
                inferred.append(f"video.{vk} <- {k}")
    return Mapping(state=state, action=action, video=video, scalars=dict(spec.get("scalars") or {}),
                   ignore=list(spec.get("ignore") or []), episodes=spec.get("episodes", "data/demo_*"),
                   fps=spec.get("fps"), task_attr=spec.get("task_attr"),
                   success_attr=spec.get("success_attr", "success"), inferred=inferred)


def _stack(ep: H5Episode, entries: list[tuple[str, str]]) -> np.ndarray | None:
    parts = []
    for k, _ in entries:
        if k not in ep.datasets:
            return None
        a = np.asarray(ep.datasets[k])
        parts.append(a.reshape(len(a), -1).astype(np.float64))
    if not parts:
        return None
    n = min(len(p) for p in parts)
    return np.concatenate([p[:n] for p in parts], axis=1)


# =====================================================================================
# Numeric comparison helpers
# =====================================================================================
def _levels(x: np.ndarray) -> int:
    x = x[np.isfinite(x)]
    return int(np.unique(x).size) if x.size else 0


def _pinned(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 20 or np.ptp(x) <= 0:
        return 0.0
    lo, hi = x.min(), x.max()
    tol = 1e-6 * (hi - lo)
    return float(np.mean((x <= lo + tol) | (x >= hi - tol)))


def _hf_share(x: np.ndarray, cut: float) -> float:
    x = x[np.isfinite(x)]
    if x.size < 32:
        return math.nan
    t = np.arange(x.size)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    p = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(x.size)
    total = p[1:].sum()
    return float(p[f > cut].sum() / total) if total > 1e-18 else math.nan


def information_metrics(src: np.ndarray, out: np.ndarray) -> dict[str, float]:
    """How much of one signal survived: error relative to spread, SNR in dB, effective bits.

    Effective bits = log2(range / (RMS error * sqrt(12))): the resolution of an
    ideal quantizer that would leave the same error. float32 storage of a
    radian-scale joint leaves ~20+ bits; 8 bits means the conversion behaves like
    an 8-bit quantizer, whatever the dtype says.
    """
    x, y = np.asarray(src, float), np.asarray(out, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return {"max_rel_error": math.nan, "snr_db": math.nan, "effective_bits": math.nan, "rmse": math.nan}
    e = y[ok] - x[ok]
    spread = float(np.std(x[ok]))
    rng = float(np.ptp(x[ok]))
    rmse = float(np.sqrt(np.mean(e ** 2)))
    scale = spread if spread > 1e-12 else max(float(np.max(np.abs(x[ok]))), 1e-12)
    return {
        "max_rel_error": float(np.max(np.abs(e))) / scale,
        "snr_db": math.inf if rmse <= 1e-15 else 10 * math.log10(max(spread ** 2, 1e-30) / rmse ** 2),
        "effective_bits": math.inf if rmse <= 1e-15 else (math.log2(rng / (rmse * math.sqrt(12))) if rng > 0 else math.nan),
        "rmse": rmse,
    }


_KNOWN_SCALES = {57.29577951308232: "radians -> degrees", 1 / 57.29577951308232: "degrees -> radians",
                 1000.0: "m -> mm (or s -> ms)", 0.001: "mm -> m (or ms -> s)", 100.0: "m -> cm or 0-1 -> 0-100",
                 0.01: "cm -> m or 0-100 -> 0-1", -1.0: "sign flipped"}


def _diagnose(src: np.ndarray, out: np.ndarray) -> str | None:
    """Explain a large per-dimension error: a known scale, a sign flip or a constant offset."""
    ok = np.isfinite(src) & np.isfinite(out)
    x, y = src[ok], out[ok]
    if x.size < 5 or np.std(x) < 1e-12:
        return None
    k, b = np.polyfit(x, y, 1)
    resid = y - (k * x + b)
    if np.std(resid) > 0.05 * max(np.std(y), 1e-12):
        return None
    for s, label in _KNOWN_SCALES.items():
        if abs(k - s) < 0.02 * abs(s) and abs(b) < 0.05 * max(np.std(y), 1e-9):
            return f"output = {k:.4g} x source ({label})"
    if abs(k - 1) < 0.01 and abs(b) > 0.05 * np.std(x):
        return f"constant offset of {b:.4g} (calibration or zeroing applied)"
    return f"output ≈ {k:.4g} x source + {b:.4g} (a scale or gain applied)"


def _permutation(src: np.ndarray, out: np.ndarray, names: list[str]) -> list[str]:
    """Columns of the output that match a *different* source column far better than their own."""
    found = []
    d = min(src.shape[1], out.shape[1])
    for j in range(d):
        y = out[:, j]
        if np.std(y) < 1e-12:
            continue
        errs = [float(np.sqrt(np.nanmean((src[:, i] - y) ** 2))) for i in range(src.shape[1])]
        best = int(np.argmin(errs))
        if best != j and errs[best] < 0.1 * max(errs[j], 1e-12):
            found.append(f"output {names[j] if j < len(names) else j} carries source {names[best] if best < len(names) else best}")
    return found


def _dim_names(entries: list[tuple[str, str]], ep: H5Episode) -> list[str]:
    names = []
    for k, n in entries:
        a = np.asarray(ep.datasets[k])
        w = int(np.prod(a.shape[1:])) if a.ndim > 1 else 1
        names += [n] if w == 1 else [f"{n}[{i}]" for i in range(w)]
    return names


def _best_lag(src: np.ndarray, out: np.ndarray, max_lag: int = 3) -> int:
    """Shift k (out[t] = src[t + k]) that minimizes error; 0 when the streams are aligned."""
    n = min(len(src), len(out))
    best, best_e = 0, math.inf
    for k in range(-max_lag, max_lag + 1):
        a = src[max(0, k): n + min(0, k)]
        b = out[max(0, -k): n - max(0, k)]
        if len(a) < 5:
            continue
        e = float(np.nanmean((a - b) ** 2))
        if e < best_e - 1e-15:
            best, best_e = k, e
    return best


# =====================================================================================
# LeRobot output
# =====================================================================================
def _open_lerobot(root: str | Path):
    from groot_eval.open_loop import LeRobotDataset

    return LeRobotDataset.open(root)


def _frames(path: Path) -> list[np.ndarray]:
    """Decode every frame as RGB. OpenCV first; if it yields nothing (OpenCV builds often lack an AV1
    decoder, and AV1 is LeRobot's default codec), fall back to the ffmpeg command line."""
    frames: list[np.ndarray] = []
    try:
        from groot_eval.open_loop import _VideoReader

        r = _VideoReader(path)
        try:
            frames = list(r.iter_frames())
        finally:
            r.close()
    except (ImportError, FileNotFoundError):
        frames = []
    return frames or _ffmpeg_frames(path)


def _ffmpeg_frames(path: Path) -> list[np.ndarray]:
    import shutil
    import subprocess

    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return []
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        w, h = (int(x) for x in probe.stdout.strip().split(",")[:2])
    except ValueError:
        return []
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    return list(np.frombuffer(raw[: n * w * h * 3], np.uint8).reshape(n, h, w, 3)) if n else []


def _to_uint8(frames: np.ndarray) -> np.ndarray:
    a = np.asarray(frames)
    if np.issubdtype(a.dtype, np.floating):
        return np.clip(np.round(a * 255.0 if np.nanmax(a) <= 1.0 + 1e-6 else a), 0, 255).astype(np.uint8)
    return a.astype(np.uint8)


# =====================================================================================
# Gates
# =====================================================================================
@dataclass
class Pair:
    index: int
    h5: H5Episode
    frame: Any  # pandas DataFrame of the LeRobot episode
    meta: dict[str, Any]


def _pairs(eps: list[H5Episode], ds: Any) -> tuple[list[Pair], list[Finding]]:
    out: list[Finding] = []
    metas = list(ds.episodes)
    by_source = {str(m.get("source_episode") or m.get("source_demo")): m for m in metas
                 if m.get("source_episode") or m.get("source_demo")}
    pairs = []
    if by_source:
        for ep in eps:
            m = by_source.get(ep.name)
            if m is None:
                out.append(Finding("error", "episodes", f"HDF5 episode {ep.name} has no LeRobot episode (source_episode)"))
                continue
            pairs.append(Pair(int(m["episode_index"]), ep, ds.episode_frame(int(m["episode_index"])), m))
    else:
        for ep, m in zip(eps, metas):
            pairs.append(Pair(int(m["episode_index"]), ep, ds.episode_frame(int(m["episode_index"])), m))
        if len(eps) != len(metas):
            out.append(Finding("error", "episodes", f"{len(eps)} HDF5 episodes, {len(metas)} LeRobot episodes "
                                                    "(paired in order; record `source_episode` in episodes.jsonl to pair by name)"))
    return pairs, out


def inventory(pairs: list[Pair], mapping: Mapping, cfg: dict[str, Any], root_attrs: dict[str, Any]) -> tuple[list[Finding], dict[str, Any]]:
    out: list[Finding] = []
    m: dict[str, Any] = {"episodes": len(pairs)}
    bad_len = [(p.h5.name, p.h5.length, len(p.frame)) for p in pairs if p.h5.length != len(p.frame)]
    if bad_len:
        out.append(Finding("error", "length", f"{len(bad_len)} episode(s) changed length (HDF5 -> LeRobot rows), e.g. "
                                              f"{bad_len[:3]}: dropped, trimmed or resampled steps"))
    used = {k for k, _ in mapping.state} | {k for k, _ in mapping.action} | set(mapping.video.values()) | set(mapping.scalars.values())
    all_keys = sorted({k for p in pairs for k in p.h5.datasets})
    ignored = [k for k in all_keys if any(fnmatch.fnmatch(k, pat) for pat in mapping.ignore)]
    lost = [k for k in all_keys if k not in used and k not in ignored]
    m.update({"converted": sorted(used & set(all_keys)), "ignored": ignored, "lost": lost})
    missing = sorted(used - set(all_keys))
    if missing:
        out.append(Finding("error", "mapping", f"mapped HDF5 keys not found in the file: {missing}"))
    if lost:
        severe = [k for k in lost if _FORCE_LIKE.search(k) or (_IMAGE_LIKE.search(k) and pairs and pairs[0].h5.datasets[k].ndim >= 3)]
        level = "error" if cfg["lost_is_error"] else "warn"
        out.append(Finding(level, "lost", f"{len(lost)} HDF5 dataset(s) not converted and not listed under `ignore`: {lost[:12]}"
                           + (f"; includes sensing a policy may need: {severe[:6]}" if severe else "")))
    if mapping.inferred:
        out.append(Finding("ok", "mapping", "inferred from modality.json: " + "; ".join(mapping.inferred[:12])))
    attrs = sorted({k for p in pairs for k in p.h5.attrs if k not in ("num_samples",)})
    kept = {k for p in pairs for k in p.meta}
    dropped = [a for a in attrs if a not in kept and a != mapping.task_attr]
    if dropped:
        out.append(Finding("warn", "attrs", f"episode attributes not carried into episodes.jsonl: {dropped[:10]}"))
    if root_attrs:
        m["source_attrs"] = sorted(root_attrs)[:20]
    if not any(f.level in ("error", "warn") for f in out):
        out.append(Finding("ok", "inventory", f"{len(pairs)} episodes, lengths match, every dataset converted or ignored"))
    return out, m


def _h5_fps(mapping: Mapping, pairs: list[Pair], root_attrs: dict[str, Any]) -> float | None:
    if mapping.fps:
        return float(mapping.fps)
    for src in [root_attrs] + [p.h5.attrs for p in pairs[:1]]:
        for k in ("fps", "control_freq", "control_frequency", "frequency", "rate_hz"):
            if isinstance(src.get(k), (int, float)) and src[k] > 0:
                return float(src[k])
        env_args = src.get("env_args")
        if isinstance(env_args, dict):
            kw = env_args.get("env_kwargs") or env_args
            for k in ("control_freq", "fps"):
                if isinstance(kw.get(k), (int, float)) and kw[k] > 0:
                    return float(kw[k])
            if isinstance(kw.get("sim_dt"), (int, float)) and isinstance(kw.get("decimation"), (int, float)):
                return 1.0 / (kw["sim_dt"] * kw["decimation"])
    for p in pairs[:1]:
        for k in p.h5.datasets:
            if k.rsplit("/", 1)[-1] in ("timestamp", "timestamps", "time"):
                t = np.asarray(p.h5.datasets[k], float).ravel()
                d = np.diff(t)
                d = d[d > 0]
                if d.size:
                    return float(1.0 / np.median(d))
    return None


def timing(pairs: list[Pair], ds: Any, mapping: Mapping, cfg: dict[str, Any], root_attrs: dict[str, Any]) -> tuple[list[Finding], dict[str, Any]]:
    out: list[Finding] = []
    fps_out = float(ds.info.get("fps") or 0) or None
    fps_src = _h5_fps(mapping, pairs, root_attrs)
    m: dict[str, Any] = {"fps_lerobot": fps_out, "fps_source": fps_src}
    if fps_out is None:
        out.append(Finding("error", "fps", "LeRobot info.json has no fps"))
    if fps_src is None:
        out.append(Finding("warn", "fps", "the HDF5 file records no control rate (fps / control_freq / sim_dt*decimation / "
                                          "timestamps); set `mapping.fps` so the rate can be checked"))
    elif fps_out and abs(fps_out - fps_src) / fps_src > cfg["max_fps_rel_error"]:
        out.append(Finding("error", "fps", f"control rate changed: source {fps_src:g} Hz, LeRobot {fps_out:g} Hz"))
    elif fps_out:
        out.append(Finding("ok", "fps", f"{fps_out:g} Hz on both sides"))
    ratios = [len(p.frame) / p.h5.length for p in pairs if p.h5.length]
    if ratios and any(abs(r - 1) > 1e-9 for r in ratios):
        r = float(np.median(ratios))
        if abs(r - 0.5) < 0.05 or abs(r - 2) < 0.1 or abs(r - 1 / 3) < 0.05:
            out.append(Finding("error", "resample", f"LeRobot episodes are {r:.2f}x the HDF5 length: resampled"))
    jit = []
    for p in pairs:
        if "timestamp" in p.frame and len(p.frame) > 2 and fps_out:
            d = np.diff(p.frame["timestamp"].to_numpy(dtype=float))
            jit.append(float(np.max(np.abs(d - 1 / fps_out)) * fps_out))
            if np.any(d <= 0):
                out.append(Finding("error", "timestamp", f"episode {p.index}: timestamps not strictly increasing"))
    if jit:
        m["period_jitter_frac_max"] = max(jit)
        out.append(Finding("error" if max(jit) > cfg["max_period_jitter_frac"] else "ok", "jitter",
                           f"timestamp period deviates from 1/fps by up to {100 * max(jit):.1f}% of a step"))
    lags = []
    src_state = [(_stack(p.h5, mapping.state), np.stack(p.frame["observation.state"].to_numpy()).astype(float))
                 for p in pairs[:10] if mapping.state and "observation.state" in p.frame]
    for s, o in src_state:
        if s is not None and s.shape[1] == o.shape[1] and len(s) > 10:
            live = np.std(s, axis=0) > 1e-9
            if live.any():
                lags.append(_best_lag(s[:, live], o[:, live]))
    if lags and any(lags):
        out.append(Finding("error", "shift", f"state stream shifted by {_most_common(lags)} step(s) relative to the source "
                                              "(off-by-one when pairing observations with actions?)"))
    for vk, hk in mapping.video.items():
        bad = []
        for p in pairs:
            path = ds.video_path(p.index, vk)
            if not path.exists():
                bad.append(f"{p.index}: missing")
                continue
            n = _frame_count(path)
            if n >= 0 and n != len(p.frame):
                bad.append(f"{p.index}: {n} frames for {len(p.frame)} rows")
        if bad:
            out.append(Finding("error", "video.frames", f"{vk}: frame count differs from row count in {len(bad)} episode(s): {bad[:3]}"))
    return out, m


def _frame_count(path: Path) -> int:
    try:
        import cv2

        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
    except ImportError:
        n = 0
    return n if n > 0 else len(_frames(path))


def _most_common(values: list[int]) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def actuator(pairs: list[Pair], mapping: Mapping, cfg: dict[str, Any]) -> tuple[list[Finding], dict[str, Any]]:
    out: list[Finding] = []
    m: dict[str, Any] = {"signals": {}}
    for column, entries in (("observation.state", mapping.state), ("action", mapping.action)):
        if not entries:
            out.append(Finding("warn", column, f"no HDF5 keys mapped to {column}; nothing compared"))
            continue
        srcs, outs = [], []
        for p in pairs:
            s = _stack(p.h5, entries)
            if s is None or column not in p.frame:
                continue
            o = np.stack(p.frame[column].to_numpy()).astype(np.float64)
            if s.shape[1] != o.shape[1]:
                out.append(Finding("error", f"{column}.width", f"episode {p.index}: source has {s.shape[1]} values per "
                                                              f"step, LeRobot {o.shape[1]} (keys {[k for k, _ in entries]})"))
                srcs = []
                break
            n = min(len(s), len(o))
            srcs.append(s[:n])
            outs.append(o[:n])
        if not srcs:
            continue
        names = _dim_names(entries, pairs[0].h5)
        S, O = np.concatenate(srcs), np.concatenate(outs)
        new_nan = int(np.sum(np.isfinite(S) & ~np.isfinite(O)))
        if new_nan:
            out.append(Finding("error", f"{column}.finite", f"conversion introduced {new_nan} NaN/Inf values"))
        perm = _permutation(S, O, names)
        if perm:
            out.append(Finding("error", f"{column}.order", "joint/dimension order changed: " + "; ".join(perm[:6])))
        dtype_src = {str(np.asarray(pairs[0].h5.datasets[k]).dtype) for k, _ in entries}
        worst_rel, worst_name = 0.0, ""
        bits = []
        for j, name in enumerate(names[: S.shape[1]]):
            info = information_metrics(S[:, j], O[:, j])
            m["signals"][f"{column}:{name}"] = info
            if math.isfinite(info["effective_bits"]):
                bits.append(info["effective_bits"])
            if info["max_rel_error"] > worst_rel:
                worst_rel, worst_name = info["max_rel_error"], name
            if info["max_rel_error"] > cfg["max_rel_error"] or info["snr_db"] < cfg["min_snr_db"]:
                why = _diagnose(S[:, j], O[:, j])
                lv_s, lv_o = _levels(S[:, j]), _levels(O[:, j])
                extra = []
                if why:
                    extra.append(why)
                if lv_s > 50 and lv_o < cfg["min_level_ratio"] * lv_s:
                    extra.append(f"quantized: {lv_o} distinct values left of {lv_s}")
                if _pinned(O[:, j]) - _pinned(S[:, j]) > cfg["max_pinned_increase"]:
                    extra.append(f"clipped: {100 * _pinned(O[:, j]):.1f}% of samples at the extremes")
                hs = np.mean([_hf_share(s[:, j], cfg["hf_cut"]) for s in srcs])
                ho = np.mean([_hf_share(o[:, j], cfg["hf_cut"]) for o in outs])
                if math.isfinite(hs) and hs > 1e-6 and math.isfinite(ho) and ho / hs < cfg["min_hf_ratio"]:
                    extra.append(f"high-frequency content cut to {100 * ho / hs:.0f}% (filtered or smoothed)")
                out.append(Finding("error", f"{column}.values",
                                   f"{name}: max error {info['max_rel_error']:.3g} of its spread, SNR {info['snr_db']:.1f} dB, "
                                   f"{info['effective_bits']:.1f} effective bits" + (f"; {'; '.join(extra)}" if extra else "")))
        dS = np.concatenate([np.diff(s, axis=0) for s in srcs if len(s) > 2])
        dO = np.concatenate([np.diff(o, axis=0) for o in outs if len(o) > 2])
        live = np.std(dS, axis=0) > 1e-12
        if live.any():
            dn = float(np.nanmax(tm.nrmse(dO[:, live], dS[:, live])))
            m[f"{column}.diff_nrmse_max"] = dn
            if dn > cfg["max_diff_nrmse"] and not any(f.check == f"{column}.values" for f in out):
                out.append(Finding("error", f"{column}.velocity", f"step-to-step changes differ by up to {dn:.3g} of their "
                                                                  "spread although values look close: jitter or reordering"))
        m[f"{column}.min_effective_bits"] = min(bits) if bits else None
        if not any(f.check.startswith(column) and f.level == "error" for f in out):
            note = (f"; source dtype {sorted(dtype_src)} stored as float32 (expected rounding only)"
                    if "float64" in dtype_src else "")
            out.append(Finding("ok", f"{column}.values", f"{len(names)} dims preserved: worst {worst_rel:.2g} of spread "
                                                         f"({worst_name}), at least {min(bits) if bits else math.inf:.1f} "
                                                         f"effective bits{note}"))
    return out, m


def visual(pairs: list[Pair], ds: Any, mapping: Mapping, cfg: dict[str, Any],
           encoder: Callable[[np.ndarray], np.ndarray] | None = None) -> tuple[list[Finding], dict[str, Any]]:
    out: list[Finding] = []
    m: dict[str, Any] = {}
    if not mapping.video:
        return [Finding("warn", "video", "no camera streams mapped; visual fidelity not checked")], m
    every = max(1, int(cfg["sample_every"]))
    cap = int(cfg["max_frames_per_episode"])
    for vk, hk in mapping.video.items():
        ps, ss, sharp, shift, frozen, offsets, emb = [], [], [], [], [], [], []
        shapes, swapped, rng_bug, checked = set(), 0, 0, 0
        for p in pairs:
            if hk not in p.h5.datasets or p.h5.datasets[hk].ndim < 3:
                out.append(Finding("error", f"{vk}.source", f"HDF5 key {hk} missing or not an image stream in {p.h5.name}"))
                continue
            path = ds.video_path(p.index, vk)
            if not path.exists():
                out.append(Finding("error", f"{vk}.file", f"episode {p.index}: {path} missing"))
                continue
            src = _to_uint8(p.h5.datasets[hk])
            raw = p.h5.datasets[hk]
            got = _frames(path)
            if not got:
                out.append(Finding("error", f"{vk}.decode", f"episode {p.index}: no frames decoded from {path}"))
                continue
            checked += 1
            shapes.add((tuple(src.shape[1:3]), tuple(got[0].shape[:2])))
            n = min(len(src), len(got))
            idx = list(range(0, n, every))[:cap]
            off = tm.best_frame_offset([src[i] for i in idx[:60]], [got[i] for i in idx[:60]], max_offset=3)
            offsets.append(off["offset"])
            if src.shape[1:3] != got[0].shape[:2]:
                import cv2

                conv = [cv2.resize(got[i], (src.shape[2], src.shape[1]), interpolation=cv2.INTER_AREA) for i in idx]
            else:
                conv = [got[i] for i in idx]
            ref = [src[i] for i in idx]
            co = tm.channel_order_check(ref[len(ref) // 2], conv[len(conv) // 2])
            swapped += int(co["swapped"])
            if np.issubdtype(raw.dtype, np.floating):
                bright_src = float(np.mean(ref[len(ref) // 2]))
                bright_out = float(np.mean(conv[len(conv) // 2]))
                if bright_src > 10 and bright_out < 0.1 * bright_src:
                    rng_bug += 1
            for a, b in zip(ref, conv):
                ps.append(tm.psnr(a, b))
                shift.append(np.abs(np.mean(a.reshape(-1, a.shape[-1]), axis=0) - np.mean(b.reshape(-1, b.shape[-1]), axis=0)).max()
                             if a.ndim == 3 else abs(float(a.mean() - b.mean())))
            step = max(1, len(ref) // 20)
            for a, b in zip(ref[::step], conv[::step]):
                ss.append(tm.ssim(a, b))
                sharp.append(tm.sharpness_ratio(a, b))
            same_out = sum(np.array_equal(got[i], got[i - 1]) for i in range(1, n))
            same_src = sum(np.array_equal(src[i], src[i - 1]) for i in range(1, n))
            frozen.append(max(0, same_out - same_src) / max(n - 1, 1))
            if encoder is not None:
                g = tm.policy_visual_gap(ref[::step], conv[::step], encoder=encoder)
                emb.append(g.get("embedding_cosine_distance_mean", math.nan))
        if not checked:
            continue
        fin = [x for x in ps if math.isfinite(x)]
        res = {
            "episodes": checked,
            "resolution": sorted(f"{a[1]}x{a[0]} -> {b[1]}x{b[0]}" for a, b in shapes),
            "psnr_mean": float(np.mean(fin)) if fin else math.inf,
            "psnr_p5": float(np.percentile(fin, 5)) if fin else math.inf,
            "psnr_min": float(np.min(fin)) if fin else math.inf,
            "ssim_mean": float(np.mean(ss)) if ss else math.nan,
            "ssim_min": float(np.min(ss)) if ss else math.nan,
            "sharpness_ratio": float(np.nanmedian(sharp)) if sharp else math.nan,
            "color_shift_max": float(np.max(shift)) if shift else 0.0,
            "frozen_frac_max": float(np.max(frozen)) if frozen else 0.0,
            "frame_offsets": sorted(set(offsets)),
        }
        if emb:
            res["embedding_distance_mean"] = float(np.nanmean(emb))
        m[vk] = res
        prefix = vk
        if any(a != b for a, b in shapes):
            out.append(Finding("error", f"{prefix}.resolution", f"resolution changed: {res['resolution']} (scaled: detail lost "
                                                               "and camera intrinsics no longer match)"))
        if swapped:
            out.append(Finding("error", f"{prefix}.channels", f"RGB/BGR order swapped in {swapped} of {checked} episode(s)"))
        if rng_bug:
            out.append(Finding("error", f"{prefix}.range", f"float [0, 1] frames came out almost black in {rng_bug} episode(s) "
                                                          "(cast to uint8 without scaling by 255)"))
        if any(o != 0 for o in offsets):
            out.append(Finding("error", f"{prefix}.alignment", f"frames shifted against the rows by {res['frame_offsets']} "
                                                              "(a dropped or extra frame at the start)"))
        checks = [
            (res["psnr_mean"] < cfg["min_psnr_mean"], f"mean PSNR {res['psnr_mean']:.1f} dB (limit {cfg['min_psnr_mean']})"),
            (res["psnr_p5"] < cfg["min_psnr_p5"], f"5th-percentile PSNR {res['psnr_p5']:.1f} dB (limit {cfg['min_psnr_p5']})"),
            (math.isfinite(res["ssim_mean"]) and res["ssim_mean"] < cfg["min_ssim_mean"],
             f"mean SSIM {res['ssim_mean']:.3f} (limit {cfg['min_ssim_mean']})"),
            (math.isfinite(res["sharpness_ratio"]) and res["sharpness_ratio"] < cfg["min_sharpness_ratio"],
             f"sharpness kept {100 * res['sharpness_ratio']:.0f}% (limit {100 * cfg['min_sharpness_ratio']:.0f}%): blur, "
             "downscaling or heavy compression"),
            (res["color_shift_max"] > cfg["max_color_shift"],
             f"mean colour shifted by up to {res['color_shift_max']:.1f} levels (limit {cfg['max_color_shift']}): colour "
             "space or range conversion (full vs limited range)"),
            (res["frozen_frac_max"] > cfg["max_frozen_frac"],
             f"{100 * res['frozen_frac_max']:.1f}% of frames repeat the previous one where the source moves (frames "
             "duplicated, or motion lost to compression)"),
            ("embedding_distance_mean" in res and res["embedding_distance_mean"] > cfg["max_embedding_distance"],
             f"policy-embedding distance {res.get('embedding_distance_mean', 0):.3f} (limit {cfg['max_embedding_distance']})"),
        ]
        bad = [msg for cond, msg in checks if cond]
        for msg in bad:
            out.append(Finding("error", f"{prefix}.quality", msg))
        if not any(f.check.startswith(prefix) and f.level == "error" for f in out):
            out.append(Finding("ok", f"{prefix}.quality", f"{checked} episode(s): PSNR {res['psnr_mean']:.1f} dB mean / "
                                                         f"{res['psnr_p5']:.1f} dB p5, SSIM {res['ssim_mean']:.3f}, "
                                                         f"sharpness {100 * res['sharpness_ratio']:.0f}%"))
    return out, m


def metadata(pairs: list[Pair], ds: Any, mapping: Mapping) -> tuple[list[Finding], dict[str, Any]]:
    out: list[Finding] = []
    empty = [p.index for p in pairs if not (p.meta.get("tasks") or [""])[0]]
    if empty:
        out.append(Finding("error", "task", f"{len(empty)} episode(s) without a task string (GR00T conditions on it): {empty[:5]}"))
    if mapping.task_attr:
        wrong = [p.index for p in pairs if str(p.h5.attrs.get(mapping.task_attr, "")) and
                 str(p.h5.attrs.get(mapping.task_attr)) not in (p.meta.get("tasks") or [])]
        if wrong:
            out.append(Finding("error", "task", f"task string differs from HDF5 attr `{mapping.task_attr}` in {len(wrong)} episode(s)"))
    if mapping.success_attr:
        src = [(p, p.h5.attrs.get(mapping.success_attr)) for p in pairs if mapping.success_attr in p.h5.attrs]
        if src:
            lost = [p.index for p, _ in src if "success" not in p.meta]
            flip = [p.index for p, v in src if "success" in p.meta and bool(p.meta["success"]) != bool(v)]
            if lost:
                out.append(Finding("warn", "success", f"success label not carried for {len(lost)} episode(s); failed demos "
                                                      "can no longer be filtered out of training"))
            if flip:
                out.append(Finding("error", "success", f"success label flipped in {len(flip)} episode(s): {flip[:5]}"))
    mod = ds.modality or {}
    if mapping.state and mod.get("state") and pairs:
        widths = {}
        for k, name in mapping.state:
            a = np.asarray(pairs[0].h5.datasets[k])
            widths[name] = int(np.prod(a.shape[1:])) if a.ndim > 1 else 1
        for name, sl in mod["state"].items():
            if name in widths and sl["end"] - sl["start"] != widths[name]:
                out.append(Finding("error", "modality", f"modality.json gives state.{name} width {sl['end'] - sl['start']}, "
                                                        f"the source key has {widths[name]}"))
        order_src = [n for _, n in mapping.state if n in mod["state"]]
        order_mod = [n for n, _ in sorted(mod["state"].items(), key=lambda kv: kv[1]["start"]) if n in order_src]
        if order_src != order_mod:
            out.append(Finding("error", "modality", f"modality.json orders state keys {order_mod}, the conversion "
                                                    f"concatenated {order_src}"))
    feats = ds.info.get("features") or {}
    for col in ("observation.state", "action"):
        shape = (feats.get(col) or {}).get("shape")
        if shape and pairs and col in pairs[0].frame:
            w = len(pairs[0].frame[col].iloc[0])
            if int(np.prod(shape)) != w:
                out.append(Finding("error", "info", f"info.json declares {col} shape {shape}, data has {w}"))
    if not mod:
        out.append(Finding("warn", "modality", "no meta/modality.json: GR00T needs it to split state and action keys"))
    if not any(f.level != "ok" for f in out):
        out.append(Finding("ok", "metadata", "tasks, labels, modality.json and info.json consistent"))
    return out, {}


# =====================================================================================
# Runner and report
# =====================================================================================
def check_conversion(h5_path: str | Path, lerobot_root: str | Path, cfg: dict[str, Any] | None = None,
                     encoder: Callable[[np.ndarray], np.ndarray] | None = None) -> list[GateResult]:
    cfg = cfg or load_config(None)
    g = cfg["gates"]
    ds = _open_lerobot(lerobot_root)
    pattern = (cfg.get("mapping") or {}).get("episodes", "data/demo_*")
    eps, root_attrs = read_hdf5(h5_path, pattern)
    results: list[GateResult] = []
    if not eps:
        return [GateResult("C0_inventory", "fail", True, [Finding("error", "episodes", f"no episode groups match {pattern!r}")])]
    mapping = build_mapping(cfg.get("mapping") or {}, eps[0], ds.modality or {})
    pairs, pf = _pairs(eps, ds)

    def add(name: str, findings: list[Finding], metrics: dict[str, Any]) -> None:
        results.append(GateResult(name, _status(findings), bool(g[name]["required"]), findings, metrics))

    f0, m0 = inventory(pairs, mapping, g["C0_inventory"], root_attrs)
    add("C0_inventory", pf + f0, m0)
    add("C1_timing", *timing(pairs, ds, mapping, g["C1_timing"], root_attrs))
    add("C2_actuator", *actuator(pairs, mapping, g["C2_actuator"]))
    add("C3_visual", *visual(pairs, ds, mapping, g["C3_visual"], encoder))
    add("C4_metadata", *metadata(pairs, ds, mapping))
    return results


def conversion_markdown(results: list[GateResult], h5_path: str, lerobot_root: str) -> str:
    code = exit_code(results)
    verdict = {0: "PASS", 1: "FAIL", 2: "INCOMPLETE"}[code]
    icon = {"pass": "pass", "warn": "pass (warnings)", "fail": "**FAIL**", "skip": "skipped"}
    lines = [f"# HDF5 -> LeRobot conversion fidelity: {verdict}", "",
             f"Source `{h5_path}`, output `{lerobot_root}`.", "",
             "| Gate | Checks | Status |", "|---|---|---|",
             *[f"| {r.gate} | {WHAT[r.gate]} | {icon[r.status]} |" for r in results]]
    sig = next((r.metrics.get("signals") for r in results if r.gate == "C2_actuator"), None)
    vis = next((r.metrics for r in results if r.gate == "C3_visual"), None)
    if sig or vis:
        lines += ["", "## Information kept", "",
                  "Actuator rows: error relative to the signal's spread, SNR and effective bits (the resolution of an ideal "
                  "quantizer leaving the same error; float32 of a radian-scale joint gives 20+). Camera rows: PSNR (dB; "
                  "40+ is visually lossless, under 30 is visible), SSIM (1 = identical) and sharpness kept.", "",
                  "| Signal | Max error / spread | SNR (dB) | Effective bits |", "|---|---|---|---|"]
        rows = sorted((sig or {}).items(), key=lambda kv: -(kv[1]["max_rel_error"] if math.isfinite(kv[1]["max_rel_error"]) else 0))
        for name, v in rows[:20]:
            lines.append(f"| {name} | {v['max_rel_error']:.2g} | {_f1(v['snr_db'])} | {_f1(v['effective_bits'])} |")
        if len(rows) > 20:
            lines.append(f"| ... {len(rows) - 20} more in the JSON | | | |")
        if vis:
            lines += ["", "| Camera | Resolution | PSNR mean / p5 (dB) | SSIM mean / min | Sharpness kept | Colour shift |",
                      "|---|---|---|---|---|---|"]
            for cam, v in vis.items():
                lines.append(f"| {cam} | {', '.join(v['resolution'])} | {_f1(v['psnr_mean'])} / {_f1(v['psnr_p5'])} | "
                             f"{v['ssim_mean']:.3f} / {v['ssim_min']:.3f} | {100 * v['sharpness_ratio']:.0f}% | "
                             f"{v['color_shift_max']:.1f} |")
    for r in results:
        lines += ["", f"## {r.gate}: {icon[r.status]}", ""]
        for f in sorted(r.findings, key=lambda f: {"error": 0, "warn": 1, "ok": 2}[f.level]):
            tag = {"error": "**ERROR**", "warn": "warn", "ok": "ok"}[f.level]
            lines.append(f"- {tag} `{f.check}`: {f.detail}")
    return "\n".join(lines) + "\n"


def _f1(x: float) -> str:
    return "inf" if x == math.inf else ("-" if not math.isfinite(x) else f"{x:.1f}")


# =====================================================================================
# Reference converter
# =====================================================================================
VIDEO_CODECS = {
    # name: ffmpeg arguments (None = OpenCV writer). "av1" mirrors LeRobot's own encoder settings.
    "h264-lossless": ["-c:v", "libx264rgb", "-crf", "0", "-preset", "veryfast"],
    "h264": ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-preset", "veryfast"],
    "av1": ["-c:v", "libsvtav1", "-crf", "30", "-g", "2", "-pix_fmt", "yuv420p"],
    "mp4v": None,
}


def write_video(frames: np.ndarray, path: Path, fps: float, codec: str = "h264-lossless") -> None:
    """Encode RGB uint8 frames (T, H, W, 3) to mp4 with ffmpeg (or OpenCV for ``mp4v``)."""
    import shutil
    import subprocess

    if codec not in VIDEO_CODECS:
        raise ValueError(f"unknown codec {codec!r}; known: {sorted(VIDEO_CODECS)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    args = VIDEO_CODECS[codec]
    if args is not None and shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
               "-r", f"{fps:g}", "-i", "-", *args, str(path)]
        subprocess.run(cmd, input=np.ascontiguousarray(frames[..., :3]).tobytes(), check=True, capture_output=True)
        return
    import cv2

    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    for fr in frames:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR) if fr.ndim == 3 else fr)
    vw.release()


def convert_hdf5_to_lerobot(h5_path: str | Path, out: str | Path, mapping_spec: dict[str, Any], fps: float,
                            task: str = "", codec: str = "h264-lossless") -> Path:
    """A small, explicit HDF5 -> LeRobot v2 converter to compare a production pipeline against.

    Writes ``observation.state`` / ``action`` from the mapped keys, one mp4 per
    camera (``codec``: ``h264-lossless`` by default, ``av1`` to mirror LeRobot's
    own encoder and see the loss it costs), ``modality.json`` from the mapping, and keeps
    ``source_episode`` plus the HDF5 episode attributes in ``episodes.jsonl``.
    """
    from groot_eval.open_loop import write_lerobot_dataset

    eps, _ = read_hdf5(h5_path, mapping_spec.get("episodes", "data/demo_*"))
    if not eps:
        raise ValueError(f"no episodes in {h5_path}")
    mp = build_mapping(mapping_spec, eps[0], {})
    payload = []
    for ep in eps:
        states, actions = _stack(ep, mp.state), _stack(ep, mp.action)
        if states is None or actions is None:
            raise ValueError(f"{ep.name}: mapped state/action keys missing")
        n = min(len(states), len(actions))
        meta = {"source_episode": ep.name, **{k: v for k, v in ep.attrs.items()
                                                if isinstance(v, (bool, int, float, str)) and k != mp.task_attr}}
        t = str(ep.attrs.get(mp.task_attr, task)) if mp.task_attr else task
        extra = {col: np.asarray(ep.datasets[k], float).reshape(len(ep.datasets[k]), -1)[:n, 0]
                 for col, k in mp.scalars.items() if k in ep.datasets}
        payload.append({"task": t, "states": states[:n], "actions": actions[:n], "meta": meta, "extra": extra})

    def slices(entries: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
        out_, start = {}, 0
        for k, name in entries:
            a = np.asarray(eps[0].datasets[k])
            w = int(np.prod(a.shape[1:])) if a.ndim > 1 else 1
            out_[name] = {"start": start, "end": start + w}
            start += w
        return out_

    modality = {"state": slices(mp.state), "action": slices(mp.action), "annotation": {"task": {}}}
    if mp.video:
        modality["video"] = {vk: {"original_key": f"observation.images.{vk}"} for vk in mp.video}
    fps_v = int(round(fps)) if abs(fps - round(fps)) < 1e-6 else float(fps)
    root = write_lerobot_dataset(out, payload, fps=fps_v, modality=modality)
    from groot_eval.open_loop import DEFAULT_VIDEO_PATH

    for i, ep in enumerate(eps):
        n = len(payload[i]["states"])
        for vk, hk in mp.video.items():
            frames = _to_uint8(ep.datasets[hk])[:n]
            path = Path(root) / DEFAULT_VIDEO_PATH.format(episode_chunk=i // 1000, video_key=vk, episode_index=i)
            write_video(frames, path, fps, codec)
    return Path(root)
