"""HDF5 -> LeRobot -> HDF5 round trip and sim replay: can converted demonstrations still be reproduced in sim?

:mod:`groot_eval.conversion_fidelity` (C0-C4) checks that the LeRobot output kept the
information GR00T trains on. This module adds the two gates that need the trip back:

====  ==========  =====================================================================
C5    roundtrip   rebuild a robomimic / Isaac Lab HDF5 from the LeRobot output plus a
                  sim sidecar, and compare it with the source: mapped streams within
                  tolerance, cameras above a PSNR bar, sim state bit-exact, shapes,
                  dtypes and attributes restored; reports what cannot be rebuilt and
                  the PSNR ceiling of the video pixel format (chroma subsampling alone)
C6    replay      replay the source and the rebuilt HDF5 in the simulator and split any
                  mismatch three ways: the sim is nondeterministic, the sim has drifted
                  since recording, or the conversion damaged actions or start states;
                  optionally re-render recorded states and compare with recorded frames
====  ==========  =====================================================================

Why a sidecar: LeRobot keeps video, state, action and task. It does not keep what
Isaac Lab needs to replay a demo: ``initial_state`` (articulations and rigid
objects), per-step ``states`` and the file's ``env_args``. The conversion mapping
usually lists them under ``ignore``. :func:`write_sidecar` copies them byte for byte
into ``<dataset>/sim_sidecar/`` with SHA-256 hashes, next to the LeRobot data;
LeRobot and GR00T readers ignore the folder. Run it right after the production
converter (``groot-twin convert-sidecar``), or pass ``--sidecar`` to ``groot-twin convert``.

Configuration: the same YAML as ``convert-check``. ``mapping`` is shared; this
module's settings live under a separate ``roundtrip`` section so ``convert-check``
keeps rejecting unknown gate names::

    roundtrip:
      sidecar: ["initial_state", "states"]
      gates:
        C5_roundtrip: {max_rel_error: 1.0e-4, min_psnr_mean: 35, min_psnr_p5: 30}
        C6_replay: {joint_abs: 1.0e-3, source_joint_abs: 1.0e-2}
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import yaml

from groot_eval import conversion_fidelity as cf
from groot_eval.fidelity import GateResult, _status, exit_code
from groot_eval.types import Finding

SIDECAR_DIR = "sim_sidecar"
MANIFEST = "manifest.json"
ROOT_ATTRS = "_root_attrs.hdf5"
DEFAULTS: dict[str, dict[str, Any]] = {
    "C5_roundtrip": {"required": True, "max_rel_error": 1e-4, "min_psnr_mean": 35.0, "min_psnr_p5": 30.0,
                     "lost_is_error": False, "ceiling_frames": 4},
    "C6_replay": {"required": False, "joint_abs": 1e-3, "floor_multiplier": 3.0, "source_joint_abs": 1e-2,
                  "min_psnr_mean": 30.0, "min_psnr_p5": 25.0},
}
WHAT = {
    "C5_roundtrip": "HDF5 rebuilt from LeRobot + sim sidecar matches the source; what cannot be rebuilt",
    "C6_replay": "source and rebuilt demos replay the same in sim; sim determinism and drift separated out",
}


def load_config(path: str | Path | None) -> dict[str, Any]:
    """``mapping`` (shared with convert-check) plus the ``roundtrip`` section."""
    cfg: dict[str, Any] = {"mapping": {}, "sidecar": ["initial_state", "states"],
                           "gates": {k: dict(v) for k, v in DEFAULTS.items()}}
    if path:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        cfg["mapping"] = data.get("mapping") or {}
        rt = data.get("roundtrip") or {}
        if "sidecar" in rt:
            cfg["sidecar"] = list(rt["sidecar"])
        for k, v in (rt.get("gates") or {}).items():
            if k not in cfg["gates"]:
                raise ValueError(f"unknown roundtrip gate {k!r}; known: {sorted(cfg['gates'])}")
            cfg["gates"][k].update(v or {})
    return cfg


def _under(path: str, roots: list[str]) -> bool:
    return any(path == r or path.startswith(r.rstrip("/") + "/") for r in roots)


def _ignored(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def _sha(arrs: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for k in sorted(arrs):
        a = np.ascontiguousarray(arrs[k])
        h.update(f"{k}|{a.dtype.str}|{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _episode_groups(f: Any, pattern: str) -> tuple[Any, list[str]]:
    parent_path, _, leaf = pattern.rpartition("/")
    parent = f[parent_path] if parent_path else f
    return parent, sorted((n for n in parent.keys() if fnmatch.fnmatch(n, leaf)), key=cf._natural)


def _leaves(grp: Any) -> dict[str, Any]:
    import h5py

    out: dict[str, Any] = {}
    grp.visititems(lambda name, obj: out.__setitem__(name, obj) if isinstance(obj, h5py.Dataset) else None)
    return out


def _pairing(names: list[str], ds: Any) -> list[tuple[str, int]]:
    by_source = {str(m.get("source_episode") or m.get("source_demo")): int(m["episode_index"])
                 for m in ds.episodes if m.get("source_episode") or m.get("source_demo")}
    if by_source:
        missing = [n for n in names if n not in by_source]
        if missing:
            raise ValueError(f"HDF5 episodes without a LeRobot episode (source_episode): {missing[:5]}")
        return [(n, by_source[n]) for n in names]
    if len(names) != len(ds.episodes):
        raise ValueError(f"{len(names)} HDF5 episodes vs {len(ds.episodes)} LeRobot episodes; record "
                         "`source_episode` in episodes.jsonl to pair by name")
    return [(n, int(m["episode_index"])) for n, m in zip(names, ds.episodes)]


# =====================================================================================
# Sidecar and reverse conversion
# =====================================================================================
def write_sidecar(h5_path: str | Path, lerobot_root: str | Path, cfg: dict[str, Any] | None = None) -> Path:
    """Copy the sim-only datasets (``cfg['sidecar']`` prefixes) and all attributes next to a LeRobot dataset."""
    import h5py

    cfg = cfg or load_config(None)
    pattern = cfg["mapping"].get("episodes", "data/demo_*")
    ds = cf._open_lerobot(lerobot_root)
    root = Path(lerobot_root) / SIDECAR_DIR
    root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"source_hdf5": str(h5_path), "episodes_pattern": pattern,
                                "sidecar": list(cfg["sidecar"]), "episodes": []}
    with h5py.File(h5_path, "r") as f:
        parent, names = _episode_groups(f, pattern)
        with h5py.File(root / ROOT_ATTRS, "w") as ra:
            for k, v in parent.attrs.items():
                ra.attrs[k] = v
        for name, idx in _pairing(names, ds):
            grp = parent[name]
            leaves = _leaves(grp)
            side = {p: d[()] for p, d in leaves.items() if _under(p, cfg["sidecar"])}
            with h5py.File(root / f"episode_{idx:06d}.hdf5", "w") as out:
                for p, arr in side.items():
                    out.create_dataset(p, data=arr)
                for k, v in grp.attrs.items():
                    out.attrs[k] = v
            manifest["episodes"].append({
                "episode_index": idx, "source_episode": name, "sha256": _sha(side),
                "leaves": {p: {"dtype": d.dtype.str, "shape": list(d.shape)} for p, d in leaves.items()},
            })
    (root / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def _width(shape: list[int]) -> int:
    return int(np.prod(shape[1:])) if len(shape) > 1 else 1


def lerobot_to_hdf5(lerobot_root: str | Path, cfg: dict[str, Any], out: str | Path) -> tuple[Path, list[Finding]]:
    """Rebuild a robomimic-layout HDF5 from LeRobot + sidecar, using only the mapping. Returns (path, findings)."""
    import h5py

    findings: list[Finding] = []
    side_root = Path(lerobot_root) / SIDECAR_DIR
    if not (side_root / MANIFEST).exists():
        raise FileNotFoundError(f"{side_root / MANIFEST} missing: run `groot-twin convert-sidecar` (or `convert --sidecar`) "
                                "when the dataset is produced; without it initial_state / states are gone and the "
                                "demos cannot be replayed")
    manifest = json.loads((side_root / MANIFEST).read_text(encoding="utf-8"))
    ds = cf._open_lerobot(lerobot_root)
    first = manifest["episodes"][0]
    sample = cf.H5Episode(first["source_episode"], {
        p: np.zeros([1] + v["shape"][1:], dtype=np.dtype(v["dtype"])) for p, v in first["leaves"].items()})
    mapping = cf.build_mapping(cfg["mapping"], sample, ds.modality or {})
    parent_path = manifest["episodes_pattern"].rpartition("/")[0]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tampered = []
    with h5py.File(out, "w") as f:
        parent = f.require_group(parent_path) if parent_path else f
        with h5py.File(side_root / ROOT_ATTRS, "r") as ra:
            for k, v in ra.attrs.items():
                parent.attrs[k] = v
        for e in manifest["episodes"]:
            idx, leaves = int(e["episode_index"]), e["leaves"]
            frame = ds.episode_frame(idx)
            n = len(frame)
            data: dict[str, np.ndarray] = {}
            for col, entries in (("observation.state", mapping.state), ("action", mapping.action)):
                if not entries or col not in frame:
                    continue
                arr = np.stack(frame[col].to_numpy()).astype(np.float64).reshape(n, -1)
                start = 0
                for key, _ in entries:
                    if key not in leaves:
                        continue
                    w = _width(leaves[key]["shape"])
                    data[key] = arr[:, start:start + w].reshape([n] + leaves[key]["shape"][1:])
                    start += w
            for col, key in mapping.scalars.items():
                if col in frame and key in leaves:
                    data[key] = np.asarray(frame[col].to_numpy(), dtype=np.float64).reshape([n] + leaves[key]["shape"][1:])
            for vk, key in mapping.video.items():
                path = ds.video_path(idx, vk)
                frames = cf._frames(path) if path.exists() else []
                if not frames:
                    findings.append(Finding("error", f"{vk}.decode", f"episode {idx}: no frames from {path}"))
                    continue
                v = np.stack(frames)
                if np.dtype(leaves[key]["dtype"]).kind == "f":  # source was float [0, 1]; cf._to_uint8 scaled it
                    v = v.astype(np.float64) / 255.0
                data[key] = v
            with h5py.File(side_root / f"episode_{idx:06d}.hdf5", "r") as s:
                side = {p: d[()] for p, d in _leaves(s).items()}
                attrs = dict(s.attrs)
            if _sha(side) != e["sha256"]:
                tampered.append(idx)
            data.update(side)
            g = parent.create_group(e["source_episode"])
            for k, v in attrs.items():
                g.attrs[k] = v
            for key, arr in data.items():
                want = np.dtype(leaves[key]["dtype"]) if key in leaves else arr.dtype
                g.create_dataset(key, data=np.asarray(arr).astype(want, copy=False))
    if tampered:
        findings.append(Finding("error", "sidecar.hash", f"sidecar changed after it was written, episodes {tampered[:5]}"))
    return out, findings


# =====================================================================================
# C5 round trip
# =====================================================================================
def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def _pix_fmt(path: Path) -> str | None:
    import shutil
    import subprocess

    if not shutil.which("ffprobe"):
        return None
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=pix_fmt",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return r.stdout.strip() or None


def colorspace_roundtrip(frames: np.ndarray, pix_fmt: str) -> np.ndarray | None:
    """RGB -> ``pix_fmt`` -> RGB through ffmpeg's scaler, no codec: the best a video in this format can be."""
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        return None
    t, h, w = frames.shape[:3]
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "-",
                        "-vf", f"format={pix_fmt},format=rgb24", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       input=np.ascontiguousarray(frames[..., :3]).tobytes(), capture_output=True)
    if r.returncode or len(r.stdout) != t * h * w * 3:
        return None
    return np.frombuffer(r.stdout, np.uint8).reshape(t, h, w, 3)


def roundtrip(orig: str | Path, recon: str | Path, lerobot_root: str | Path, cfg: dict[str, Any],
              pre: list[Finding] | None = None) -> GateResult:
    g = cfg["gates"]["C5_roundtrip"]
    pattern = cfg["mapping"].get("episodes", "data/demo_*")
    a_eps, a_root = cf.read_hdf5(orig, pattern)
    b_eps, b_root = cf.read_hdf5(recon, pattern)
    ds = cf._open_lerobot(lerobot_root)
    out: list[Finding] = list(pre or [])
    m: dict[str, Any] = {"signals": {}, "cameras": {}}
    if not a_eps:
        return GateResult("C5_roundtrip", "fail", bool(g["required"]), [Finding("error", "episodes", f"no {pattern} in {orig}")])
    mapping = cf.build_mapping(cfg["mapping"], a_eps[0], ds.modality or {})
    b_by = {e.name: e for e in b_eps}
    mapped = {k for k, _ in mapping.state + mapping.action} | set(mapping.scalars.values())
    videos = {hk: vk for vk, hk in mapping.video.items()}
    sidecar = list(cfg["sidecar"])
    missing_eps = [e.name for e in a_eps if e.name not in b_by]
    if missing_eps:
        out.append(Finding("error", "episodes", f"not rebuilt: {missing_eps[:5]}"))
    errs: dict[str, float] = {}
    psnrs: dict[str, list[float]] = {}
    side_bad, shape_bad, attr_bad = set(), set(), []
    ignored, lost = set(), set()
    for a in a_eps:
        b = b_by.get(a.name)
        if b is None:
            continue
        if {k: str(v) for k, v in a.attrs.items()} != {k: str(v) for k, v in b.attrs.items()}:
            attr_bad.append(a.name)
        for key, x in a.datasets.items():
            if key not in b.datasets:
                (ignored if _ignored(key, mapping.ignore) else lost).add(key)
                continue
            y = b.datasets[key]
            if x.shape != y.shape or x.dtype != y.dtype:
                if key not in videos or x.shape[1:] != y.shape[1:]:
                    shape_bad.add(f"{key} {x.dtype}{list(x.shape)} -> {y.dtype}{list(y.shape)}")
            if _under(key, sidecar):
                if x.shape != y.shape or not np.array_equal(x, y):
                    side_bad.add(key)
            elif key in videos:
                n = min(len(x), len(y))
                if len(x) != len(y):
                    shape_bad.add(f"{key}: {len(y)} frames vs {len(x)}")
                xs, ys = cf._to_uint8(x[:n]), cf._to_uint8(y[:n])
                if xs.shape[1:] == ys.shape[1:]:
                    psnrs.setdefault(key, []).extend(min(_psnr(p, q), 100.0) for p, q in zip(xs, ys))
            elif key in mapped:
                xf, yf = x.astype(np.float64).reshape(len(x), -1), y.astype(np.float64).reshape(len(y), -1)
                if xf.shape == yf.shape:
                    spread = np.ptp(xf, axis=0)
                    spread[spread <= 0] = 1.0
                    errs[key] = max(errs.get(key, 0.0), float(np.max(np.abs(xf - yf) / spread)) if xf.size else 0.0)
    if [str(v) for v in a_root.values()] != [str(b_root.get(k)) for k in a_root]:
        out.append(Finding("error", "attrs.root", f"file attributes differ: "
                           f"{[k for k in a_root if str(a_root[k]) != str(b_root.get(k))][:5]}"))
    if attr_bad:
        out.append(Finding("error", "attrs.episode", f"episode attributes differ in {attr_bad[:5]}"))
    for key, e in sorted(errs.items()):
        m["signals"][key] = {"max_rel_error": e}
        out.append(Finding("error" if e > g["max_rel_error"] else "ok", f"{key}.values",
                           f"max error {e:.3g} of the signal's spread after the round trip (limit {g['max_rel_error']:g})"))
    for key, ps in sorted(psnrs.items()):
        arr = np.array(ps)
        mean, p5 = float(arr.mean()), float(np.percentile(arr, 5))
        m["cameras"][key] = {"psnr_mean": mean, "psnr_p5": p5}
        bad = mean < g["min_psnr_mean"] or p5 < g["min_psnr_p5"]
        out.append(Finding("error" if bad else "ok", f"{videos[key]}.psnr",
                           f"rebuilt frames vs source: PSNR mean {mean:.1f} dB, p5 {p5:.1f} dB "
                           f"(limits {g['min_psnr_mean']:g} / {g['min_psnr_p5']:g})"))
    out += _ceilings(a_eps, ds, mapping, g, m)
    if side_bad:
        out.append(Finding("error", "sidecar", f"sim state not bit-exact after the round trip: {sorted(side_bad)[:5]}"))
    elif any(_under(k, sidecar) for e in a_eps for k in e.datasets):
        out.append(Finding("ok", "sidecar", f"sim state ({', '.join(sidecar)}) restored bit-exact"))
    else:
        out.append(Finding("warn", "sidecar", f"source has no {sidecar} datasets: the rebuilt demos cannot be replayed "
                                              "from their recorded start state"))
    if shape_bad:
        out.append(Finding("error", "shape", f"shape or dtype not restored: {sorted(shape_bad)[:5]}"))
    if ignored:
        out.append(Finding("warn", "ignored", f"{len(ignored)} dataset(s) listed under `ignore` are in neither LeRobot "
                                              f"nor the sidecar and cannot be rebuilt: {sorted(ignored)[:6]}"))
    if lost:
        out.append(Finding("error" if g["lost_is_error"] else "warn", "lost",
                           f"{len(lost)} dataset(s) neither mapped, ignored nor in the sidecar: {sorted(lost)[:6]}"))
    m["ignored"], m["lost"] = sorted(ignored), sorted(lost)
    return GateResult("C5_roundtrip", _status(out), bool(g["required"]), out, m)


def _ceilings(eps: list[Any], ds: Any, mapping: Any, g: dict[str, Any], m: dict[str, Any]) -> list[Finding]:
    """Split visual loss into pixel format (chroma subsampling) and codec, so a failure says which to change."""
    out = []
    for vk, hk in mapping.video.items():
        ep = eps[0]
        path = ds.video_path(0, vk)
        if hk not in ep.datasets or not path.exists():
            continue
        fmt = _pix_fmt(path)
        if not fmt:
            continue
        if fmt.startswith(("rgb", "bgr", "gbr", "yuv444")):
            out.append(Finding("ok", f"{vk}.pix_fmt", f"{fmt}: no chroma subsampling"))
            continue
        src = cf._to_uint8(ep.datasets[hk])
        idx = np.unique(np.linspace(0, len(src) - 1, min(len(src), int(g["ceiling_frames"]))).round().astype(int))
        back = colorspace_roundtrip(src[idx], fmt)
        if back is None:
            continue
        ceil = float(np.mean([min(_psnr(a, b), 100.0) for a, b in zip(src[idx], back)]))
        m["cameras"].setdefault(hk, {})["pix_fmt_ceiling_db"] = ceil
        bad = ceil < g["min_psnr_mean"]
        out.append(Finding("error" if bad else "ok", f"{vk}.pix_fmt",
                           f"{fmt} alone caps PSNR at {ceil:.1f} dB before any compression"
                           + (f" (< {g['min_psnr_mean']:g}): no codec setting can pass; small saturated features lose "
                              "colour. Encode yuv444p or rgb, or lower the bar knowingly" if bad else "")))
    return out


def check_roundtrip(h5_path: str | Path, lerobot_root: str | Path, cfg: dict[str, Any], recon: str | Path) -> GateResult:
    """Rebuild ``recon`` from LeRobot + sidecar and run C5."""
    g = cfg["gates"]["C5_roundtrip"]
    try:
        _, pre = lerobot_to_hdf5(lerobot_root, cfg, recon)
    except FileNotFoundError as exc:
        return GateResult("C5_roundtrip", "fail", bool(g["required"]), [Finding("error", "sidecar", str(exc))])
    return roundtrip(h5_path, recon, lerobot_root, cfg, pre)


# =====================================================================================
# C6 replay
# =====================================================================================
@dataclass
class ReplayResult:
    joint_pos: np.ndarray  # (T, D) joint positions after each step
    frames: dict[str, np.ndarray] = field(default_factory=dict)  # HDF5 camera key -> (T, H, W, 3)
    success: bool | None = None


class DemoReplayer(Protocol):
    """Plays one HDF5 demo in a simulator: ``actions`` open loop from the initial state, or ``states`` teleported."""

    def replay(self, h5_path: str, demo: str, mode: Literal["actions", "states"], render: bool = False) -> ReplayResult: ...


def _div(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    n = min(len(a), len(b))
    e = np.abs(np.asarray(a[:n], float) - np.asarray(b[:n], float)).reshape(n, -1)
    over = e.max(axis=1) > 1e-3 if e.size else np.zeros(0, bool)
    return {"max": float(e.max()) if e.size else 0.0, "rms": float(np.sqrt((e ** 2).mean())) if e.size else 0.0,
            "first_diverge_step": int(np.argmax(over)) if over.any() else -1, "len_a": len(a), "len_b": len(b)}


def check_replay(h5_path: str | Path, recon: str | Path, cfg: dict[str, Any], replayer: DemoReplayer,
                 joint_key: str | None = None, render: bool = True, demos: list[str] | None = None) -> GateResult:
    g = cfg["gates"]["C6_replay"]
    pattern = cfg["mapping"].get("episodes", "data/demo_*")
    eps, _ = cf.read_hdf5(h5_path, pattern)
    mapping = cf.build_mapping(cfg["mapping"], eps[0], {}) if eps else None
    out: list[Finding] = []
    m: dict[str, Any] = {}
    for ep in eps:
        if demos and ep.name not in demos:
            continue
        a1 = replayer.replay(str(h5_path), ep.name, "actions")
        a2 = replayer.replay(str(h5_path), ep.name, "actions")
        floor = _div(a1.joint_pos, a2.joint_pos)
        noisy = floor["max"] > g["joint_abs"]
        out.append(Finding("warn" if noisy else "ok", f"{ep.name}.determinism",
                           f"source replayed twice: max joint difference {floor['max']:.2e}"
                           + ("; the sim is nondeterministic, the conversion limit is widened to match" if noisy else "")))
        rec = {"determinism": floor}
        if joint_key and joint_key in ep.datasets:
            # recorded obs at t is before the step; replayed joints at t are after it
            src = _div(np.asarray(ep.datasets[joint_key], float)[1:], a1.joint_pos[:-1])
            rec["source"] = src
            bad = src["max"] > g["source_joint_abs"]
            out.append(Finding("error" if bad else "ok", f"{ep.name}.source",
                               f"source replay vs recorded {joint_key}: max {src['max']:.2e}"
                               + (f", diverges at step {src['first_diverge_step']}: the source does not reproduce in this "
                                  "sim build (Isaac Sim / Lab version, USD assets, physics dt, decimation); not a "
                                  "conversion problem" if bad else "")))
        r = replayer.replay(str(recon), ep.name, "actions")
        conv = _div(a1.joint_pos, r.joint_pos)
        allowed = max(g["joint_abs"], g["floor_multiplier"] * floor["max"])
        bad = conv["max"] > allowed or conv["len_a"] != conv["len_b"]
        rec["conversion"] = {**conv, "allowed": allowed}
        out.append(Finding("error" if bad else "ok", f"{ep.name}.conversion",
                           f"rebuilt vs source replay: max joint difference {conv['max']:.2e} (limit {allowed:.1e})"
                           + (f", first at step {conv['first_diverge_step']}" if bad else "")))
        if a1.success is not None and r.success is not None:
            if a1.success != r.success:
                out.append(Finding("error", f"{ep.name}.success", f"task success {a1.success} from the source, "
                                                                   f"{r.success} from the rebuilt demo"))
            elif not a1.success:
                out.append(Finding("warn", f"{ep.name}.success", "the source demo does not succeed when replayed: "
                                                                 "exclude it or re-record"))
        if render and mapping and mapping.video:
            s = replayer.replay(str(recon), ep.name, "states", render=True)
            for vk, hk in mapping.video.items():
                if hk not in s.frames or hk not in ep.datasets:
                    continue
                x, y = cf._to_uint8(ep.datasets[hk]), cf._to_uint8(s.frames[hk])
                n = min(len(x), len(y))
                if x.shape[1:] != y.shape[1:]:
                    out.append(Finding("error", f"{ep.name}.{vk}.render", f"re-rendered {y.shape[1:3]} vs recorded {x.shape[1:3]}"))
                    continue
                ps = np.array([min(_psnr(p, q), 100.0) for p, q in zip(x[:n], y[:n])])
                mean, p5 = float(ps.mean()), float(np.percentile(ps, 5))
                rec[f"render.{vk}"] = {"psnr_mean": mean, "psnr_p5": p5}
                bad = mean < g["min_psnr_mean"] or p5 < g["min_psnr_p5"]
                out.append(Finding("error" if bad else "ok", f"{ep.name}.{vk}.render",
                                   f"recorded states re-rendered vs recorded frames: PSNR mean {mean:.1f}, p5 {p5:.1f} dB"
                                   + ("; the sim does not reproduce its own visuals (renderer, lighting, assets, camera "
                                      "pose or channel order changed)" if bad else "")))
        m[ep.name] = rec
    return GateResult("C6_replay", _status(out), bool(g["required"]), out, m)


# =====================================================================================
# Report
# =====================================================================================
def roundtrip_markdown(results: list[GateResult], h5_path: str, lerobot_root: str) -> str:
    code = exit_code(results)
    icon = {"pass": "pass", "warn": "pass (warnings)", "fail": "**FAIL**", "skip": "skipped"}
    lines = [f"# HDF5 -> LeRobot -> HDF5 round trip: {({0: 'PASS', 1: 'FAIL', 2: 'INCOMPLETE'})[code]}", "",
             f"Source `{h5_path}`, LeRobot `{lerobot_root}`.", "",
             "| Gate | Checks | Status |", "|---|---|---|",
             *[f"| {r.gate} | {WHAT[r.gate]} | {icon[r.status]} |" for r in results]]
    for r in results:
        lines += ["", f"## {r.gate}: {icon[r.status]}", ""]
        for f in sorted(r.findings, key=lambda f: {"error": 0, "warn": 1, "ok": 2}[f.level]):
            tag = {"error": "**ERROR**", "warn": "warn", "ok": "ok"}[f.level]
            lines.append(f"- {tag} `{f.check}`: {f.detail}")
    return "\n".join(lines) + "\n"
