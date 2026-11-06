"""Run manifests: everything that can change a result, written next to it.

Two runs are only comparable if they used the same checkpoint, simulator
version, config and latency model. A regression gate that silently compares a
run on Isaac Sim 4.5 against a baseline from 4.2, or checkpoint step 30k
against 32k, reports noise as a regression (or hides a real one). Every
``sweep`` and ``regress`` writes ``manifest.json``; ``compare`` and
``regress`` diff manifests and warn (or, with ``--strict``, fail) on any
field in :data:`RESULT_FIELDS` that differs.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1
# Fields whose difference can change outcomes. Host details are recorded but only reported.
RESULT_FIELDS = ("checkpoint.sha256", "policy", "policy_args", "env", "env_args", "config_sha256",
                 "latency", "packages.isaacsim", "packages.isaaclab", "packages.gr00t", "packages.torch",
                 "packages.numpy", "groot_eval.git_sha", "groot_eval.lock_sha256", "gpu")
_PACKAGES = ("numpy", "pandas", "pyarrow", "torch", "isaacsim", "isaaclab", "gr00t", "gymnasium")
_ROOT = Path(__file__).resolve().parents[2]


def sha256_path(path: str | Path, chunk: int = 1 << 23) -> str:
    """Content hash of a file, or of a directory (relative paths + contents, sorted)."""
    path = Path(path)
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    if not files:
        raise ValueError(f"{path}: nothing to hash")
    for f in files:
        if path.is_dir():
            h.update(f.relative_to(path).as_posix().encode() + b"\0")
        with open(f, "rb") as fh:
            while block := fh.read(chunk):
                h.update(block)
    return h.hexdigest()


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", str(_ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


def _packages() -> dict[str, str]:
    out = {}
    for name in _PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def _gpu() -> str | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip().replace("\n", "; ") or None if r.returncode == 0 else None


def build_manifest(*, policy: str, policy_args: dict[str, Any] | None = None, env: str = "",
                   env_args: dict[str, Any] | None = None, config: str | Path | None = None,
                   checkpoint: str | Path | None = None, latency: str | None = None,
                   extra: dict[str, Any] | None = None) -> dict[str, Any]:
    from groot_eval import __version__

    lock = _ROOT / "uv.lock"
    m: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "policy": policy,
        "policy_args": policy_args or {},
        "env": env,
        "env_args": env_args or {},
        "config_sha256": sha256_path(config) if config else None,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_path(checkpoint)} if checkpoint else None,
        "latency": latency or "none (sim waits for the policy)",
        "groot_eval": {"version": __version__, "git_sha": _git_sha(),
                       "lock_sha256": sha256_path(lock) if lock.exists() else None},
        "packages": _packages(),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "gpu": _gpu(),
        "argv": sys.argv[1:],
    }
    if extra:
        m.update(extra)
    return m


def unpinned(m: dict[str, Any]) -> list[str]:
    """Reasons this run cannot be reproduced exactly."""
    out = []
    if not m.get("checkpoint") and ("groot" in str(m.get("policy", "")) or ":" in str(m.get("policy", ""))):
        out.append("no --checkpoint given: the model behind the policy is not pinned by hash")
    if str(m.get("latency", "")).startswith("measured"):
        out.append("latency is `measured`: it depends on machine load, so outcomes can change between runs")
    if str((m.get("groot_eval") or {}).get("git_sha") or "").endswith("+dirty"):
        out.append("harness code has uncommitted changes")
    return out


def _get(m: dict[str, Any], dotted: str) -> Any:
    cur: Any = m
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def diff_manifests(a: dict[str, Any] | None, b: dict[str, Any] | None, *, ignore: tuple[str, ...] = ()) -> list[str]:
    """Human-readable differences on result-relevant fields (``a`` = baseline, ``b`` = candidate)."""
    if not a or not b:
        return ["one side has no manifest: cannot tell whether checkpoint, simulator or config changed"]
    out = []
    for f in RESULT_FIELDS:
        if f in ignore:
            continue
        va, vb = _get(a, f), _get(b, f)
        if va != vb:
            out.append(f"{f}: {json.dumps(va, default=str)} -> {json.dumps(vb, default=str)}")
    return out


def write_manifest(m: dict[str, Any], out_dir: str | Path) -> Path:
    path = Path(out_dir) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    return path


def load_manifest(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    p = p / "manifest.json" if p.is_dir() else p
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
