"""Hardware bridge: run the digital twin next to the real robot, tick by tick.

Two modes:

* **shadow**: the robot runs its own controller. Each tick the bridge reads
  the robot's state and the command the robot applied, steps the twin with
  that same command, and compares the twin's next state with the robot's.
  Nothing is sent to the robot.
* **mirror**: the bridge sends the commands. They come from a policy (closed
  loop on the robot's observations) or from a recorded trace (open-loop
  replay of captured data on the hardware). The same commands step the twin.
  With ``command_mask`` the bridge drives only some action dimensions (an arm
  of a humanoid, say) and the robot's own controller keeps the rest.

A watchdog checks every tick for twin/robot divergence, stale or missing robot
samples, robot state outside limits, command jumps and non-finite values. In
mirror mode a trip calls the transport's ``safe_stop`` and ends the episode; in
shadow mode it marks the episode (and stops only if ``stop_on_trip``).

The watchdog is a test-rig guard, not a safety system: the robot's own e-stop,
limits and safety controller stay in charge.

Both sides are recorded as traces (``robot`` with domain ``real`` and
``twin`` with domain ``sim``). The report reuses the existing tools: the
step-by-step comparison is scored like :mod:`groot_eval.replay` (nRMSE
against the robot's spread, error by steps since sync), G5 of
:mod:`groot_eval.fidelity` replays the robot's recorded commands in a fresh
twin, and every tripped episode is written as a failure record that
``groot-twin repro`` reads.

Transports: :class:`SimRobot` (an env standing in for the robot, for tests and
dry runs), :class:`TraceRobot` (a recorded robot trace, for offline shadow) and
:class:`groot_eval.bridge_ros2.Ros2Robot` (ROS 2, optional ``rclpy``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import yaml

from groot_eval.trace import TraceRecorder, TraceSet
from groot_eval.types import Scenario

Obs = dict[str, Any]
MODES = ("shadow", "mirror")
HORIZONS = (1, 5, 10, 20, 40, 80)


# =====================================================================================
# Transport
# =====================================================================================
@dataclass
class RobotSample:
    """One tick from the robot: its state after the last command, and that command as applied."""

    stamp: float  # robot clock, s
    obs: Obs  # state.* keys
    command: np.ndarray | None = None  # applied command that led to this state (None on the first sample)
    done: bool = False  # the robot side ended the episode
    info: dict[str, Any] = field(default_factory=dict)  # success / failure_reason when done


class RobotTransport(Protocol):
    """What the bridge needs from a robot connection."""

    supports_mirror: bool

    def start(self, scenario: Scenario) -> RobotSample:
        """Begin an episode and return the robot's current state."""

    def tick(self, command: np.ndarray | None, mask: np.ndarray | None, timeout_s: float) -> RobotSample | None:
        """Send ``command`` (mirror; ``mask`` = dims the bridge drives) and wait for the next sample.

        Returns ``None`` if no sample arrived within ``timeout_s``.
        """

    def safe_stop(self, reason: str, how: str) -> None:
        """Bring the robot to a safe state (``hold`` position or ``stop``)."""

    def close(self) -> None: ...


def scenarios_from_meta(meta: list[dict[str, Any]]) -> list[Scenario]:
    """The scenario each recorded episode ran (seed = episode index when the trace has none)."""
    out = []
    for i, m in enumerate(meta):
        sc = m.get("scenario") or {}
        out.append(Scenario(seed=int(sc.get("seed", i)), params=dict(sc.get("params") or {}),
                            task=str(sc.get("task") or (m.get("tasks") or [""])[0])))
    return out


class SimRobot:
    """An env standing in for the robot. Give it different parameters than the twin to model a sim/real gap.

    ``policy`` is the robot's own controller: it drives every dimension in shadow mode and the dimensions
    outside ``mask`` in mirror mode. ``drop`` lists tick indices at which the robot sends nothing (to test the
    watchdog).
    """

    supports_mirror = True

    def __init__(self, env: Any, policy: Any = None, params: dict[str, Any] | None = None,
                 drop: Sequence[int] = (), dt: float | None = None) -> None:
        self.env, self.policy, self.params = env, policy, dict(params or {})
        self.drop = set(int(d) for d in drop)
        self.dt = float(dt if dt is not None else getattr(env, "dt", None) or 0.02)
        self.stops: list[tuple[str, str]] = []
        self._obs: Obs = {}
        self._k = 0

    def start(self, scenario: Scenario) -> RobotSample:
        sc = Scenario(seed=scenario.seed, params={**scenario.params, **self.params}, task=scenario.task,
                      name=scenario.name)
        self._obs = self.env.reset(sc)
        if self.policy is not None:
            self.policy.reset(scenario.task)
        self._k = 0
        return RobotSample(0.0, dict(self._obs))

    def tick(self, command: np.ndarray | None, mask: np.ndarray | None, timeout_s: float) -> RobotSample | None:
        own = np.asarray(self.policy.act(self._obs), dtype=float) if self.policy is not None else None
        if command is None:
            if own is None:
                raise ValueError("SimRobot in shadow mode needs its own policy")
            applied = own
        else:
            applied = np.asarray(command, dtype=float).copy()
            if mask is not None:
                applied = np.where(mask, applied, own if own is not None else 0.0)
        self._k += 1
        self._obs, _, term, trunc, info = self.env.step(applied)
        if self._k in self.drop:
            return None
        done = bool(term or trunc)
        return RobotSample(self._k * self.dt, dict(self._obs), applied, done, dict(info) if done else {})

    def safe_stop(self, reason: str, how: str) -> None:
        self.stops.append((reason, how))

    def close(self) -> None:
        self.env.close()


class TraceRobot:
    """A recorded robot trace played back as the robot stream (offline shadow mode)."""

    supports_mirror = False

    def __init__(self, root: str | Path) -> None:
        self.ts = TraceSet.open(root)
        self.slices = self.ts.state_slices()
        self.states = self.ts.per_episode("observation.state")
        self.actions = self.ts.per_episode("action")
        self.meta = self.ts.episode_meta()
        frames = self.ts.frames()
        dt = 1.0 / float(self.ts.fps or 30.0)
        self.stamps = [f["timestamp"].to_numpy(dtype=float) if "timestamp" in f else np.arange(len(f)) * dt
                       for f in frames]
        self.episode = -1
        self._t = 0

    def scenarios(self) -> list[Scenario]:
        return scenarios_from_meta(self.meta)

    def _obs(self, t: int) -> Obs:
        v = self.states[self.episode][t]
        return {f"state.{k}": np.asarray(v[s], dtype=np.float32) for k, s in self.slices.items()}

    def start(self, scenario: Scenario) -> RobotSample:
        self.episode += 1
        if self.episode >= len(self.states):
            raise IndexError("the trace has no more episodes")
        self._t = 0
        return RobotSample(float(self.stamps[self.episode][0]), self._obs(0))

    def tick(self, command: np.ndarray | None, mask: np.ndarray | None, timeout_s: float) -> RobotSample | None:
        if command is not None:
            raise ValueError("a recorded trace cannot be commanded; use shadow mode")
        T = len(self.states[self.episode])
        cmd = np.asarray(self.actions[self.episode][self._t], dtype=float)
        self._t += 1
        done = self._t >= T - 1
        info: dict[str, Any] = {}
        if done:
            m = self.meta[self.episode]
            if "success" in m:
                info = {"success": bool(m["success"]), "failure_reason": m.get("failure_reason")}
        return RobotSample(float(self.stamps[self.episode][self._t]), self._obs(self._t), cmd, done, info)

    def safe_stop(self, reason: str, how: str) -> None:
        pass

    def close(self) -> None:
        pass


# =====================================================================================
# Command sources (mirror mode)
# =====================================================================================
class PolicySource:
    """Commands from a policy acting on the robot's observations."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy

    def reset(self, scenario: Scenario, episode: int) -> None:
        self.policy.reset(scenario.task)

    def next(self, obs: Obs) -> np.ndarray | None:
        return np.asarray(self.policy.act(obs), dtype=float)


class TraceSource:
    """Commands replayed from a recorded trace, open loop (one recorded episode per bridge episode)."""

    def __init__(self, root: str | Path) -> None:
        ts = TraceSet.open(root)
        self.actions = ts.per_episode("action")
        self.meta = ts.episode_meta()
        self._ep, self._t = 0, 0

    def scenarios(self) -> list[Scenario]:
        return scenarios_from_meta(self.meta)

    def reset(self, scenario: Scenario, episode: int) -> None:
        if episode >= len(self.actions):
            raise IndexError(f"the command trace has {len(self.actions)} episodes")
        self._ep, self._t = episode, 0

    def next(self, obs: Obs) -> np.ndarray | None:
        acts = self.actions[self._ep]
        if self._t >= len(acts):
            return None
        a = np.asarray(acts[self._t], dtype=float)
        self._t += 1
        return a


# =====================================================================================
# Config and watchdog
# =====================================================================================
@dataclass
class WatchdogConfig:
    max_error: dict[str, float] = field(default_factory=lambda: {"joint_pos": 0.25})  # |twin - robot| per element
    consecutive: int = 3  # ticks over max_error before tripping
    max_missing: int = 3  # consecutive ticks without a robot sample
    max_stale_s: float = 0.25  # gap in the robot clock
    state_limits: dict[str, list[float]] = field(default_factory=dict)  # robot state outside [lo, hi]
    max_command_step: float | None = None  # largest command change per tick (mirror)
    on_trip: str = "hold"  # hold | stop (what safe_stop is asked to do)
    stop_on_trip: bool | None = None  # default: True in mirror, False in shadow


@dataclass
class BridgeConfig:
    mode: str = "shadow"
    resync_every: int | None = None  # put the twin back into the robot's state every N ticks
    compare_exclude: list[str] = field(default_factory=list)  # state keys not compared
    command_mask: list[int] | None = None  # mirror: action dims the bridge drives (default all)
    max_ticks: int = 1000
    tick_timeout_s: float = 0.1
    watchdog: WatchdogConfig = field(default_factory=WatchdogConfig)

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> BridgeConfig:
        data: dict[str, Any] = {}
        if path:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        data.update({k: v for k, v in overrides.items() if v is not None})
        wd = data.pop("watchdog", {}) or {}
        known = {f for f in cls.__dataclass_fields__ if f != "watchdog"}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown bridge config keys {sorted(unknown)}; known: {sorted(known | {'watchdog'})}")
        wunknown = set(wd) - set(WatchdogConfig.__dataclass_fields__)
        if wunknown:
            raise ValueError(f"unknown watchdog keys {sorted(wunknown)}")
        cfg = cls(**data, watchdog=WatchdogConfig(**wd))
        if cfg.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {cfg.mode!r}")
        if cfg.watchdog.on_trip not in ("hold", "stop"):
            raise ValueError("watchdog.on_trip must be hold or stop")
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trip:
    tick: int
    reason: str  # divergence | missing | stale | state_limit | command_step | non_finite
    detail: str


class Watchdog:
    def __init__(self, cfg: WatchdogConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self.over = 0
        self.missing = 0
        self.last_stamp: float | None = None
        self.last_command: np.ndarray | None = None

    def missing_tick(self, tick: int) -> Trip | None:
        self.missing += 1
        if self.missing >= self.cfg.max_missing:
            return Trip(tick, "missing", f"no robot sample for {self.missing} consecutive ticks")
        return None

    def check_command(self, tick: int, command: np.ndarray) -> Trip | None:
        if not np.all(np.isfinite(command)):
            return Trip(tick, "non_finite", "non-finite command")
        lim = self.cfg.max_command_step
        if lim is not None and self.last_command is not None:
            jump = float(np.max(np.abs(command - self.last_command)))
            if jump > lim:
                return Trip(tick, "command_step", f"command jumped by {jump:.3g} (limit {lim:g})")
        self.last_command = command.copy()
        return None

    def check_sample(self, tick: int, sample: RobotSample, errors: dict[str, float]) -> Trip | None:
        self.missing = 0
        if self.last_stamp is not None and sample.stamp - self.last_stamp > self.cfg.max_stale_s:
            gap = sample.stamp - self.last_stamp
            self.last_stamp = sample.stamp
            return Trip(tick, "stale", f"robot clock jumped {1000 * gap:.0f} ms (limit {1000 * self.cfg.max_stale_s:.0f} ms)")
        self.last_stamp = sample.stamp
        for key, v in sample.obs.items():
            if key.startswith("state.") and not np.all(np.isfinite(np.asarray(v, dtype=float))):
                return Trip(tick, "non_finite", f"robot reported non-finite {key}")
        for name, (lo, hi) in self.cfg.state_limits.items():
            v = np.asarray(sample.obs.get(f"state.{name}", []), dtype=float)
            if v.size and (np.any(v < lo) or np.any(v > hi)):
                return Trip(tick, "state_limit", f"robot {name} outside [{lo}, {hi}]: {np.round(v, 3).tolist()}")
        over = [f"{k} {errors[k]:.3g} > {lim:g}" for k, lim in self.cfg.max_error.items()
                if k in errors and errors[k] > lim]
        self.over = self.over + 1 if over else 0
        if self.over >= self.cfg.consecutive:
            return Trip(tick, "divergence", f"twin and robot apart for {self.over} ticks: " + ", ".join(over))
        return None


# =====================================================================================
# Session
# =====================================================================================
def _state_keys(obs: Obs) -> list[str]:
    return sorted(k for k in obs if k.startswith("state."))


def _vec(obs: Obs, key: str) -> np.ndarray:
    return np.asarray(obs[key], dtype=np.float64).reshape(-1)


@dataclass
class EpisodeLog:
    scenario: Scenario
    keys: list[str]
    sizes: dict[str, int]  # elements per compared signal
    robot: list[np.ndarray] = field(default_factory=list)  # (D,) flattened robot state per tick
    twin: list[np.ndarray] = field(default_factory=list)
    commands: list[np.ndarray] = field(default_factory=list)
    stamps: list[float] = field(default_factory=list)
    since_sync: list[int] = field(default_factory=list)
    errors: list[dict[str, float]] = field(default_factory=list)  # max |twin - robot| per signal
    missing: int = 0
    trip: Trip | None = None
    robot_result: dict[str, Any] = field(default_factory=dict)
    twin_ended_at: int | None = None


class BridgeSession:
    """Drive one robot transport and one twin env side by side."""

    def __init__(self, robot: RobotTransport, twin: Any, cfg: BridgeConfig, source: Any = None,
                 robot_name: str = "robot", twin_name: str = "twin", control_dt: float | None = None,
                 on_tick: Callable[[int, dict[str, float]], None] | None = None) -> None:
        if cfg.mode == "mirror":
            if source is None:
                raise ValueError("mirror mode needs a command source (a policy or a recorded trace)")
            if not robot.supports_mirror:
                raise ValueError(f"{type(robot).__name__} cannot be commanded; use shadow mode")
        self.robot, self.twin, self.cfg, self.source = robot, twin, cfg, source
        self.dog = Watchdog(cfg.watchdog)
        self.on_tick = on_tick
        dt = control_dt or getattr(twin, "dt", None)
        self.rec_robot = TraceRecorder(domain="real", env=robot_name, policy=f"bridge-{cfg.mode}", control_dt=dt)
        self.rec_twin = TraceRecorder(domain="sim", env=twin_name, policy=f"bridge-{cfg.mode}", control_dt=dt)
        self.logs: list[EpisodeLog] = []

    @property
    def stop_on_trip(self) -> bool:
        s = self.cfg.watchdog.stop_on_trip
        return (self.cfg.mode == "mirror") if s is None else bool(s)

    def _sync(self, obs: Obs, recent: list[np.ndarray]) -> Obs | None:
        try:
            past = np.stack(recent) if recent else None
            out: Obs = self.twin.set_state({k: v for k, v in obs.items() if k.startswith("state.")}, past)
            return out
        except (AttributeError, NotImplementedError):
            return None

    def run_episode(self, scenario: Scenario, index: int = 0) -> EpisodeLog:
        cfg = self.cfg
        sample = self.robot.start(scenario)
        twin_obs = self.twin.reset(scenario)
        if hasattr(self.twin, "max_steps"):
            self.twin.max_steps = max(int(self.twin.max_steps), cfg.max_ticks + 1)
        synced = self._sync(sample.obs, [])
        if synced is not None:
            twin_obs = synced
        if self.source is not None:
            self.source.reset(scenario, index)
        self.dog.reset()
        self.dog.last_stamp = sample.stamp
        keys = [k for k in _state_keys(sample.obs) if k in twin_obs and k.removeprefix("state.") not in cfg.compare_exclude]
        if not keys:
            raise ValueError(f"robot and twin share no state keys (robot: {_state_keys(sample.obs)}, twin: {_state_keys(twin_obs)})")
        log = EpisodeLog(scenario, keys, {k: int(_vec(sample.obs, k).size) for k in keys})
        mask = None
        if cfg.command_mask is not None:
            dim = int(getattr(self.twin, "action_dim", max(cfg.command_mask) + 1))
            mask = np.zeros(dim, dtype=bool)
            mask[list(cfg.command_mask)] = True
        meta = {"scenario": scenario.to_dict(), "bridge_mode": cfg.mode}
        self.rec_robot.start_episode(scenario.task, **meta)
        self.rec_twin.start_episode(scenario.task, **meta)
        robot_obs, since, tick = sample.obs, 0, 0
        log.robot.append(np.concatenate([_vec(robot_obs, k) for k in keys]))
        log.twin.append(np.concatenate([_vec(twin_obs, k) for k in keys]))
        log.stamps.append(sample.stamp)
        log.since_sync.append(0)
        log.errors.append(self._errors(robot_obs, twin_obs, keys))
        trip: Trip | None = None
        twin_done = False
        while tick < cfg.max_ticks:
            command = None
            if cfg.mode == "mirror":
                command = self.source.next(robot_obs)
                if command is None:
                    break  # the recorded commands ran out
                trip = self.dog.check_command(tick, command)
                if trip:
                    break
            nxt = self.robot.tick(command, mask, cfg.tick_timeout_s)
            tick += 1
            if nxt is None:
                log.missing += 1
                trip = self.dog.missing_tick(tick)
                if trip:
                    break
                continue
            applied = nxt.command if nxt.command is not None else command
            if applied is None:
                raise ValueError("the robot sample carries no applied command; shadow mode needs one")
            applied = np.asarray(applied, dtype=float)
            twin_before = twin_obs
            if not twin_done:
                twin_obs, _, term, trunc, _ = self.twin.step(applied)
                since += 1
                if term or trunc:
                    twin_done, log.twin_ended_at = True, tick
            if cfg.resync_every and since >= cfg.resync_every:
                synced = self._sync(nxt.obs, log.commands[-8:] + [applied])
                if synced is not None:
                    twin_obs, since = synced, 0
            errs = self._errors(nxt.obs, twin_obs, keys)
            # the trace convention: the state the command was computed from, then the command
            extra = {f"bridge.err.{k.removeprefix('state.')}": v for k, v in errs.items()}
            self.rec_robot.record_step(robot_obs, applied, timestamp=log.stamps[-1], extra=extra)
            self.rec_twin.record_step(twin_before, applied, timestamp=log.stamps[-1])
            log.commands.append(applied)
            log.robot.append(np.concatenate([_vec(nxt.obs, k) for k in keys]))
            log.twin.append(np.concatenate([_vec(twin_obs, k) for k in keys]))
            log.stamps.append(nxt.stamp)
            log.since_sync.append(since)
            log.errors.append(errs)
            if self.on_tick:
                self.on_tick(tick, errs)
            robot_obs = nxt.obs
            trip = self.dog.check_sample(tick, nxt, errs)
            if trip and self.stop_on_trip:
                break
            if trip:
                log.trip = log.trip or trip  # shadow: keep the first trip and keep watching
                trip = None
            if nxt.done:
                log.robot_result = dict(nxt.info)
                break
        if trip is not None:
            log.trip = trip
            if cfg.mode == "mirror":
                self.robot.safe_stop(f"{trip.reason}: {trip.detail}", cfg.watchdog.on_trip)
        if log.trip is not None:
            self.rec_robot.end_episode(success=False, failure_reason=f"watchdog:{log.trip.reason}",
                                       watchdog=asdict(log.trip))
            self.rec_twin.end_episode(success=False, failure_reason=f"watchdog:{log.trip.reason}")
        else:
            ok = log.robot_result.get("success")
            self.rec_robot.end_episode(success=ok, failure_reason=log.robot_result.get("failure_reason") if ok is False else None)
            self.rec_twin.end_episode(success=ok)
        self.logs.append(log)
        return log

    @staticmethod
    def _errors(robot: Obs, twin: Obs, keys: list[str]) -> dict[str, float]:
        out = {}
        for k in keys:
            d = _vec(twin, k) - _vec(robot, k)
            out[k.removeprefix("state.")] = float(np.max(np.abs(d))) if d.size else 0.0
        return out

    def run(self, scenarios: Sequence[Scenario]) -> list[EpisodeLog]:
        for i, sc in enumerate(scenarios):
            log = self.run_episode(sc, i)
            if log.trip is not None and self.cfg.mode == "mirror":
                break  # a mirror trip ends the session: a person looks at the robot before it moves again
        return self.logs

    def write(self, out: str | Path) -> tuple[Path, Path]:
        out = Path(out)
        return self.rec_robot.write(out / "robot"), self.rec_twin.write(out / "twin")


# =====================================================================================
# Report
# =====================================================================================
def compare(logs: list[EpisodeLog]) -> dict[str, Any]:
    """Score the live comparison like ``replay``: per-signal nRMSE and error by steps since sync."""
    if not logs:
        return {"episodes": 0}
    keys = logs[0].keys
    slices, start = {}, 0
    for k in keys:
        n = logs[0].sizes[k]
        slices[k.removeprefix("state.")] = slice(start, start + n)
        start += n
    R = np.concatenate([np.stack(lg.robot) for lg in logs])
    scale = R.std(axis=0)
    moving = np.max([np.stack(lg.robot).std(axis=0) for lg in logs], axis=0) > 1e-9
    scale = np.where(moving, scale, 0.0)
    err = np.concatenate([(np.stack(lg.twin) - np.stack(lg.robot))[1:] for lg in logs])
    since = np.concatenate([np.asarray(lg.since_sync)[1:] for lg in logs])
    live = scale > 1e-9
    norm = np.where(live, np.abs(err) / np.where(live, scale, 1.0), np.nan)
    per_signal = {}
    for name, s in slices.items():
        cols = norm[:, s][:, live[s]]
        vals = cols[np.isfinite(cols)]
        if vals.size:
            per_signal[name] = float(np.sqrt(np.mean(vals ** 2)))
    horizons = {}
    for h in HORIZONS:
        vals = norm[since == h]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            horizons[h] = float(np.mean(vals))
    finite = [v for v in per_signal.values() if math.isfinite(v)]
    return {"episodes": len(logs), "ticks": int(len(err)), "nrmse": float(np.mean(finite)) if finite else math.nan,
            "nrmse_by_signal": per_signal, "horizons": horizons,
            "max_abs_error": {name: float(np.nanmax(np.abs(err[:, s]))) if len(err) else 0.0
                              for name, s in slices.items()}}


def step_table(log: EpisodeLog, every: int = 10, window: int = 5) -> pd.DataFrame:
    """Every tick of one episode: robot clock, steps since sync and max |twin - robot| per signal."""
    rows = []
    for t, (stamp, since, errs) in enumerate(zip(log.stamps, log.since_sync, log.errors)):
        worst = max(errs, key=lambda k: errs[k]) if errs else ""
        rows.append({"tick": t, "stamp": stamp, "since_sync": since, "worst": worst,
                     **{f"err.{k}": v for k, v in errs.items()}})
    df = pd.DataFrame(rows)
    focus = log.trip.tick if log.trip else None
    df["shown"] = (df["tick"] % max(1, every) == 0) | (df["tick"] == len(df) - 1)
    if focus is not None:
        df.loc[(df["tick"] - focus).abs() <= window, "shown"] = True
    return df


def repro_records(robot_trace: Path, logs: list[EpisodeLog], out_dir: Path) -> list[dict[str, Any]]:
    """Failure records (``groot-twin repro`` format) for every episode the watchdog tripped.

    Paths are relative to ``out_dir``, where the records file is written, as ``repro.load_failures`` expects.
    """
    try:
        rel = robot_trace.resolve().relative_to(out_dir.resolve())
    except ValueError:
        rel = robot_trace.resolve()
    recs, idx = [], 0
    for log in logs:
        if not log.commands:
            continue  # the recorder drops episodes without a step, so they have no index in the trace
        if log.trip is not None:
            recs.append({"trace": str(rel), "episode_index": idx, "domain": "real",
                         "failure_reason": f"watchdog:{log.trip.reason}", "scenario": log.scenario.to_dict(),
                         "watchdog": asdict(log.trip)})
        idx += 1
    return recs


def run_g5(robot_trace: Path, make_twin: Callable[[], Any], params: dict[str, Any] | None) -> dict[str, Any]:
    """G5 from :mod:`groot_eval.fidelity`: the robot's recorded commands replayed in a fresh twin."""
    from groot_eval.fidelity import dynamics, load_gates

    gates, _ = load_gates(None)
    env = make_twin()
    try:
        findings, metrics = dynamics(TraceSet.open(robot_trace), env, gates["G5_dynamics"], params)
    finally:
        env.close()
    status = "fail" if any(f.level == "error" for f in findings) else "warn" if any(f.level == "warn" for f in findings) else "pass"
    return {"status": status, "findings": [f.to_dict() for f in findings], "metrics": metrics}


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def bridge_markdown(result: dict[str, Any]) -> str:
    cfg = result["config"]
    c = result["comparison"]
    lines = ["# Hardware bridge report", "",
             f"- Mode: **{cfg['mode']}**; robot `{result['robot']}`, twin `{result['twin']}`"
             + (f", commands from `{result['source']}`" if result.get("source") else ""),
             f"- {c.get('episodes', 0)} episode(s), {c.get('ticks', 0)} compared ticks; missing robot samples: "
             f"{result['missing']}" + (f"; twin resynced every {cfg['resync_every']} ticks" if cfg.get("resync_every") else ""),
             ""]
    trips = [e for e in result["episodes"] if e.get("trip")]
    if trips:
        lines += [f"**Watchdog tripped in {len(trips)} episode(s).**", "", "| Episode | Tick | Reason | Detail |",
                  "|---|---|---|---|", *[f"| {e['index']} | {e['trip']['tick']} | {e['trip']['reason']} | "
                                         f"{e['trip']['detail']} |" for e in trips], ""]
    else:
        lines += ["Watchdog: no trips.", ""]
    if c.get("nrmse_by_signal"):
        lines += ["## Twin vs robot, tick by tick", "",
                  f"Overall nRMSE {c['nrmse']:.3f} (error over the robot's spread; 0 = identical).", "",
                  "| Signal | nRMSE | Max abs error |", "|---|---|---|",
                  *[f"| {k} | {v:.3f} | {c['max_abs_error'].get(k, math.nan):.4g} |"
                    for k, v in sorted(c["nrmse_by_signal"].items(), key=lambda kv: -kv[1])], ""]
    if c.get("horizons"):
        lines += ["| Steps since sync | " + " | ".join(str(h) for h in c["horizons"]) + " |",
                  "|---|" + "---|" * len(c["horizons"]),
                  "| Mean normalized error | " + " | ".join(f"{v:.3f}" for v in c["horizons"].values()) + " |", ""]
    for e in result["episodes"]:
        if e.get("steps"):
            lines += [f"### Episode {e['index']} (seed {e['seed']})" + (f": tripped at tick {e['trip']['tick']}" if e.get("trip") else ""),
                      "", "| Tick | Robot clock (s) | Since sync | Worst signal | " + " | ".join(e["signals"]) + " |",
                      "|---|---|---|---|" + "---|" * len(e["signals"]),
                      *[f"| {r['tick']} | {r['stamp']:.3f} | {r['since_sync']} | {r['worst']} | "
                        + " | ".join(f"{r['err.' + s]:.4g}" for s in e["signals"]) + " |" for r in e["steps"]], ""]
    g5 = result.get("g5")
    if g5:
        lines += ["## G5 dynamics (fidelity gate)", "", f"Status: **{g5['status']}**", "",
                  *[f"- {f['level']}: {f['detail']}" for f in g5["findings"]], ""]
    if result.get("repro_records"):
        lines += ["## Reproduce in the twin", "",
                  f"{len(result['repro_records'])} tripped episode(s) written to `repro_cases.json`. Run:", "",
                  "```bash", f"groot-twin repro {result['repro_path']} --env {result['twin_spec']} --policy <policy>", "```", ""]
    if result.get("repro"):
        r = result["repro"]
        lines += [f"Reproduction rate in the twin: {_pct(r['reproduction_rate'])} "
                  f"({r['verdicts']}); see `repro.md`.", ""]
    lines += ["The watchdog is a test-rig guard. The robot's own e-stop, limits and safety controller stay in charge.", ""]
    return "\n".join(lines)


def build_report(session: BridgeSession, out: str | Path, *, robot: str, twin_spec: str, source: str | None = None,
                 make_twin: Callable[[], Any] | None = None, twin_params: dict[str, Any] | None = None,
                 g5: bool = True, every: int = 10, run_repro: dict[str, Any] | None = None) -> dict[str, Any]:
    """Write both traces, the step-by-step CSVs, the G5 check, repro records and ``bridge.md`` / ``bridge.json``."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    robot_root, _ = session.write(out)
    episodes = []
    for i, log in enumerate(session.logs):
        df = step_table(log, every)
        df.to_csv(out / f"steps_episode{i}.csv", index=False)
        signals = [c.removeprefix("err.") for c in df.columns if c.startswith("err.")]
        episodes.append({"index": i, "seed": log.scenario.seed, "ticks": len(log.errors) - 1, "missing": log.missing,
                         "trip": asdict(log.trip) if log.trip else None, "robot_result": log.robot_result,
                         "twin_ended_at": log.twin_ended_at, "signals": signals,
                         "steps": df[df["shown"]].drop(columns="shown").to_dict("records")})
    result: dict[str, Any] = {"config": session.cfg.to_dict(), "robot": robot, "twin": twin_spec,
                              "twin_spec": twin_spec, "source": source,
                              "missing": sum(lg.missing for lg in session.logs),
                              "comparison": compare(session.logs) if session.logs else {"episodes": 0},
                              "episodes": episodes}
    if g5 and make_twin is not None and session.rec_robot.episodes:
        result["g5"] = run_g5(robot_root, make_twin, twin_params)
    recs = repro_records(robot_root, session.logs, out)
    if recs:
        p = out / "repro_cases.json"
        p.write_text(json.dumps(recs, indent=2), encoding="utf-8")
        result["repro_records"], result["repro_path"] = recs, str(p)
        if run_repro is not None:
            import importlib

            repro: Any = importlib.import_module("groot_eval.repro")  # the twin-failure-repro module
            rcfg = repro.load_config(run_repro.get("config"), env=twin_spec.split(":")[0],
                                     policy=run_repro.get("policy"))
            summary = repro.reproduce_batch(repro.load_failures(p), rcfg)
            (out / "repro.md").write_text(repro.repro_markdown(summary, rcfg), encoding="utf-8")
            (out / "repro.json").write_text(repro.to_json(summary), encoding="utf-8")
            result["repro"] = {"reproduction_rate": summary["reproduction_rate"], "verdicts": summary["verdicts"]}
    (out / "bridge.json").write_text(json.dumps(_plain(result), indent=2), encoding="utf-8")
    (out / "bridge.md").write_text(bridge_markdown(result), encoding="utf-8")
    return result


def _plain(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


# =====================================================================================
# Building from CLI specs
# =====================================================================================
def parse_env_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """``arm3`` or ``arm3:friction=1.6,action_delay=2`` -> (name, params)."""
    name, _, rest = spec.partition(":")
    params: dict[str, Any] = {}
    for item in filter(None, rest.split(",")):
        k, _, v = item.partition("=")
        params[k.strip()] = yaml.safe_load(v)
    return name, params


def make_transport(spec: str, robot_policy: str | None = None) -> tuple[RobotTransport, list[Scenario] | None]:
    """``sim:<env>[:k=v,...]``, ``trace:<dir>`` or ``ros2:<config.yaml>``."""
    kind, _, rest = spec.partition(":")
    if kind == "sim":
        from groot_eval.envs import make_env
        from groot_eval.policies import make_policy

        name, params = parse_env_spec(rest)
        return SimRobot(make_env(name), make_policy(robot_policy) if robot_policy else None, params), None
    if kind == "trace":
        tr = TraceRobot(rest)
        return tr, tr.scenarios()
    if kind == "ros2":
        from groot_eval.bridge_ros2 import Ros2Robot

        return Ros2Robot.from_config(rest), None
    raise ValueError(f"robot spec must be sim:<env>, trace:<dir> or ros2:<config.yaml>, got {spec!r}")
