"""Failure triage: one synthetic robot failure per cause, each must be attributed correctly."""

import json

import numpy as np
import pytest

from groot_eval.envs import make_env
from groot_eval.latency import with_latency
from groot_eval.policies import make_policy
from groot_eval.repro import FailureCase
from groot_eval.triage import affine_map, best_lag, stale_match, triage_case
from groot_eval.types import Scenario

KEYS = ["position", "velocity", "target"]
SLICES = {"position": slice(0, 2), "velocity": slice(2, 4), "target": slice(4, 6)}
FAST = {"kp": 6.0, "kd": 0.5}


def record(real_params, policy, scenario, map_action=None, max_steps=200):
    """Run a stand-in robot (Reach2D with real_params) and record what a robot log would hold."""
    env = make_env("reach2d")
    env.max_steps = max_steps
    obs = env.reset(Scenario(seed=scenario.seed, params={**scenario.params, **real_params}))
    policy.reset("")
    S, A, info = [], [], {}
    for _ in range(max_steps):
        S.append(np.concatenate([np.asarray(obs[f"state.{k}"], float) for k in KEYS]))
        a = np.asarray(policy.act(obs), float)
        sent = map_action(a) if map_action else a
        A.append(sent)
        obs, _, term, trunc, info = env.step(sent)
        if term or trunc:
            break
    reason = "success" if info.get("success") else info.get("failure_reason", "unknown")
    return FailureCase("c", scenario, reason, "record", "real", np.array(S), np.array(A), dict(SLICES))


CASES = {
    "policy": (lambda: record({}, make_policy("pd"), Scenario(seed=3, params={"obstacle": True})), {}),
    "plumbing": (lambda: record({}, make_policy("pd"), Scenario(seed=3), map_action=lambda a: np.array([a[1], a[0]]) * 0.6), {}),
    "timing-inference": (lambda: record({}, with_latency(make_policy("pd", **FAST), "fixed:200", 0.05), Scenario(seed=3)), FAST),
    "timing-actuation": (lambda: record({"action_delay": 3}, make_policy("pd", **FAST), Scenario(seed=3)), FAST),
    "actuation": (lambda: record({"actuator_gain": 0.4, "mass": 2.0}, make_policy("pd"), Scenario(seed=3)), {}),
    "sensing": (lambda: record({"sensor_noise": 0.08}, make_policy("pd"), Scenario(seed=3)), {}),
    "not_modelled": (lambda: record({"obstacle": True}, make_policy("pd"), Scenario(seed=3)), {}),
}


@pytest.mark.parametrize("name", list(CASES))
def test_triage_names_the_cause(name):
    make_case, policy_args = CASES[name]
    r = triage_case(make_case(), make_env("reach2d"), make_policy("pd", **policy_args))
    assert r["cause"] == name.split("-")[0], (r["cause"], r["evidence"])
    assert r["evidence"] and r["next_step"]


def test_inference_latency_is_confirmed_in_the_twin():
    make_case, args = CASES["timing-inference"]
    r = triage_case(make_case(), make_env("reach2d"), make_policy("pd", **args))
    assert r["confidence"] == "high"
    assert r["tests"]["t3_closed_loop"]["with_latency"]["finding"] == "reproduced"


def test_consistent_policy_rules_out_plumbing_and_timing():
    make_case, _ = CASES["policy"]
    r = triage_case(make_case(), make_env("reach2d"), make_policy("pd"))
    assert any(x.startswith("plumbing") for x in r["ruled_out"])
    assert any(x.startswith("timing") for x in r["ruled_out"])


def test_without_policy_triage_still_uses_replay():
    make_case, _ = CASES["timing-actuation"]
    r = triage_case(make_case(), make_env("reach2d"), None)
    assert r["tests"]["t1_consistency"]["checkable"] is False
    assert r["cause"] == "timing"


def test_helpers():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(60, 2))
    rec = np.vstack([np.zeros((2, 2)), pred[:-2]])
    assert best_lag(pred, rec, 5)[0] == 2
    m = affine_map(pred, np.c_[pred[:, 1] * 0.5, pred[:, 0]])
    assert m["order"] == [1, 0] and not m["identity"] and m["r2_min"] > 0.999
    assert affine_map(pred, pred)["identity"]
    held = np.repeat(pred[::3], 3, axis=0)[:60]
    st = stale_match(pred, np.vstack([np.zeros((3, 2)), held[:-3]]), 8, 0.01)
    assert st["matched_share"] > 0.8 and st["late_share"] > 0.5


def test_cli_triage_on_a_trace(tmp_path):
    from groot_eval.trace import TraceRecorder
    from groot_eval.twin_cli import main

    env = make_env("reach2d")
    pol = make_policy("pd")
    rec = TraceRecorder(domain="real", env="reach2d", policy="pd", control_dt=env.dt)
    sc = Scenario(seed=3)
    obs = env.reset(sc)
    rec.start_episode(env.task, scenario=sc.to_dict())
    info = {}
    for _ in range(200):
        a = np.array([pol.act(obs)[1], pol.act(obs)[0]]) * 0.6  # a robot with swapped joints
        rec.record_step(obs, a)
        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    rec.end_episode(success=False, failure_reason=info.get("failure_reason", "timeout"))
    root = rec.write(tmp_path / "real")
    assert main(["triage", str(root), "--env", "reach2d", "--policy", "pd", "--out", str(tmp_path / "out")]) == 0
    summary = json.loads((tmp_path / "out" / "triage.json").read_text())
    assert summary["results"][0]["cause"] == "plumbing"
    assert "plumbing" in (tmp_path / "out" / "triage.md").read_text()
