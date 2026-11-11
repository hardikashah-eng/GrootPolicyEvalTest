"""One run seed, many named random streams: repeatable sim2real and real2sim.

A run has a single integer **run seed**. Every source of randomness asks for its own stream by name
(``RunSeed(7).rng("faults", episode)``), and the stream's seed is a hash of (run seed, name, keys). Three
things follow:

* **Same seed, same run.** The same scenarios, wear units, fault and perturbation schedules, latency and
  sensor-noise draws, and rig stream faults, whatever the order in which components ask for them.
* **Independent streams.** Adding a new random source, or drawing more from one stream, does not shift the
  others. A plain shared ``np.random`` generator cannot promise that.
* **The seed travels with the data.** :func:`episode_meta` stamps the run seed and the derived seeds into
  each trace episode. :meth:`RunSeed.from_trace` reads them back, so a replay, repro or calibration of a
  real recording draws exactly what the robot's run drew.

:func:`fingerprint` hashes what a run did (states, actions, outcomes; not wall-clock latency).
:func:`determinism_check` runs a seed twice and finds the first step where the runs differ. Run it before
comparing sim with real: a gap that also appears between two sim runs of the same seed is non-determinism,
not a sim/real gap.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Stream names used across groot_eval. Others are allowed; these are the ones the built-in components use.
STREAMS = ("scenario", "wear_unit", "faults", "latency", "sensor_noise", "scheduler", "treadmill", "rig")
_MASK = (1 << 63) - 1


@dataclass(frozen=True)
class RunSeed:
    value: int

    def derive(self, name: str, *keys: Any) -> int:
        """A 63-bit seed for stream ``name`` (and optional keys such as an episode index or a topic)."""
        blob = json.dumps([int(self.value), str(name), [str(k) for k in keys]], separators=(",", ":"))
        return int.from_bytes(hashlib.sha256(blob.encode()).digest()[:8], "big") & _MASK

    def rng(self, name: str, *keys: Any) -> np.random.Generator:
        return np.random.default_rng(self.derive(name, *keys))

    def scenario_seeds(self, n: int) -> list[int]:
        """Seeds for ``n`` scenarios; scenario ``i`` keeps its seed when ``n`` grows."""
        return [self.derive("scenario", i) % (2 ** 31) for i in range(n)]

    def streams(self, *keys: Any, names: Sequence[str] = STREAMS) -> dict[str, int]:
        return {n: self.derive(n, *keys) for n in names}

    @classmethod
    def from_trace(cls, root: str | Path, episode: int = 0) -> RunSeed:
        """The run seed recorded in a trace episode (see :func:`episode_meta`)."""
        from groot_eval.trace import TraceSet

        meta = TraceSet.open(root).episode_meta()
        if episode >= len(meta):
            raise IndexError(f"{root} has {len(meta)} episodes")
        seed = (meta[episode].get("run_seed") or {}).get("value")
        if seed is None:
            raise KeyError(f"{root} episode {episode} carries no run_seed; it was recorded without one")
        return cls(int(seed))


def episode_meta(run: RunSeed, episode: int, names: Sequence[str] = STREAMS) -> dict[str, Any]:
    """What to pass to ``TraceRecorder.start_episode(**...)`` so the trace carries its seeds."""
    return {"run_seed": {"value": run.value, "episode": episode, "streams": run.streams(episode, names=names)}}


# =====================================================================================
# Fingerprints and determinism
# =====================================================================================
def _hash_array(h: Any, a: np.ndarray, decimals: int) -> None:
    arr = np.round(np.asarray(a, dtype=np.float64), decimals)
    arr[arr == 0] = 0.0  # -0.0 and 0.0 hash the same
    h.update(str(arr.shape).encode())
    h.update(np.nan_to_num(arr, nan=1e300).tobytes())


def episode_fingerprints(root: str | Path, decimals: int = 6) -> list[str]:
    """One hash per episode over its recorded states, actions and outcome (wall-clock columns are left out)."""
    from groot_eval.trace import TraceSet

    ts = TraceSet.open(root)
    out = []
    for meta, states, actions in zip(ts.episode_meta(), ts.per_episode("observation.state"), ts.per_episode("action")):
        h = hashlib.sha256()
        _hash_array(h, states, decimals)
        _hash_array(h, actions, decimals)
        h.update(json.dumps([meta.get("success"), meta.get("failure_reason")]).encode())
        out.append(h.hexdigest())
    return out


def fingerprint(root: str | Path, decimals: int = 6) -> str:
    return hashlib.sha256("".join(episode_fingerprints(root, decimals)).encode()).hexdigest()


def first_difference(a: str | Path, b: str | Path, atol: float = 1e-6) -> list[dict[str, Any]]:
    """For each episode: the first step at which two traces' states or actions differ (None if identical)."""
    from groot_eval.trace import TraceSet

    ta, tb = TraceSet.open(a), TraceSet.open(b)
    out = []
    for i, (sa, sb, aa, ab) in enumerate(zip(ta.per_episode("observation.state"), tb.per_episode("observation.state"),
                                             ta.per_episode("action"), tb.per_episode("action"))):
        n = min(len(sa), len(sb))
        diff = ~np.isclose(sa[:n], sb[:n], atol=atol, equal_nan=True).all(axis=1)
        diff |= ~np.isclose(aa[:n], ab[:n], atol=atol, equal_nan=True).all(axis=1)
        step = int(np.argmax(diff)) if diff.any() else (n if len(sa) != len(sb) else None)
        worst = None
        if step is not None and step < n:
            slices = ta.state_slices()
            d = np.abs(sa[step] - sb[step])
            worst = max(slices, key=lambda k: float(np.nanmax(d[slices[k]])) if d[slices[k]].size else 0.0)
        out.append({"episode": i, "first_step": step, "length": [len(sa), len(sb)], "signal": worst})
    return out


@dataclass
class DeterminismReport:
    seed: int
    runs: int
    fingerprints: list[str]
    episodes: int
    differences: list[dict[str, Any]] = field(default_factory=list)

    @property
    def deterministic(self) -> bool:
        return len(set(self.fingerprints)) == 1

    def markdown(self, title: str = "") -> str:
        lines = [f"# Determinism check{': ' + title if title else ''}", "",
                 f"Run seed {self.seed}, {self.runs} runs of {self.episodes} episodes.", ""]
        if self.deterministic:
            lines.append(f"**Deterministic**: every run has fingerprint `{self.fingerprints[0][:16]}`.")
        else:
            lines += ["**Not deterministic.** Runs of the same seed differ, so any sim/real comparison will "
                      "include this run-to-run noise.", "", "| Episode | First differing step | Lengths | Signal |",
                      "|---|---|---|---|"]
            lines += [f"| {d['episode']} | {d['first_step']} | {d['length']} | {d['signal'] or '-'} |"
                      for d in self.differences if d["first_step"] is not None]
        return "\n".join(lines) + "\n"


def determinism_check(run: Callable[[RunSeed, Path], Path], seed: int, workdir: str | Path,
                      runs: int = 2) -> DeterminismReport:
    """Call ``run(RunSeed(seed), out_dir) -> trace_root`` ``runs`` times and compare the traces."""
    workdir = Path(workdir)
    roots = [run(RunSeed(seed), workdir / f"run{i}") for i in range(runs)]
    fps = [fingerprint(r) for r in roots]
    eps = len(episode_fingerprints(roots[0]))
    diffs = first_difference(roots[0], roots[1]) if len(set(fps)) > 1 else []
    return DeterminismReport(seed, runs, fps, eps, diffs)

