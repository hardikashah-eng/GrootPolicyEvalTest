"""Failure reproduction against a synthetic 'robot': Reach2D with hidden physics the twin does not know."""

import json

import numpy as np

from groot_eval import repro
from groot_eval.envs import make_env
from groot_eval.open_loop import write_lerobot_dataset
from groot_eval.policies import make_policy
from groot_eval.types import Scenario

HIDDEN = {"actuator_gain": 3.0, "action_delay": 5}  # the 'robot' is stronger and laggier than the twin
MODALITY = {"state": {"position": {"start": 0, "end": 2}, "velocity": {"start": 2, "end": 4},
                      "target": {"start": 4, "end": 6}},
            "action": {"end_effector": {"start": 0, "end": 2}}, "annotation": {"task": {}}}


def _record_robot_failures(root, seeds=range(8)):
    env, policy = make_env("reach2d"), make_policy("pd")
    eps = []
    for seed in seeds:
        obs = env.reset(Scenario(seed=seed, params=HIDDEN))
        policy.reset("reach")
        states, actions, info = [], [], {}
        for _ in range(env.max_steps):
            act = np.asarray(policy.act(obs), dtype=float)
            states.append(np.concatenate([obs["state.position"], obs["state.velocity"], obs["state.target"]]))
            actions.append(act)
            obs, _, term, trunc, info = env.step(act)
            if term or trunc:
                break
        if not info.get("success"):
            # the robot log knows the seed but not the hidden physics
            eps.append({"task": "reach", "states": np.stack(states), "actions": np.stack(actions),
                        "meta": {"success": False, "failure_reason": info.get("failure_reason", "unknown"),
                                 "scenario": {"seed": seed, "params": {}, "task": "reach"}}})
    assert eps, "the hidden physics should make the reference controller fail"
    write_lerobot_dataset(root, eps, fps=20, modality=MODALITY)
    return len(eps)


def test_load_failures_from_trace_suite_and_record(tmp_path):
    n = _record_robot_failures(tmp_path / "robot")
    cases = repro.load_failures(tmp_path / "robot", domain="real")
    assert len(cases) == n and all(c.has_recording and c.domain == "real" for c in cases)
    suite = {"version": 1, "cases": [{"origin": "failure", "failure_reason": "collision",
                                      "scenario": {"seed": 3, "params": {"obstacle": True}}},
                                     {"origin": "anchor", "scenario": {"seed": 4}}]}
    (tmp_path / "suite.json").write_text(json.dumps(suite))
    sc = repro.load_failures(tmp_path / "suite.json")
    assert len(sc) == 1 and sc[0].failure_reason == "collision"
    rec = {"trace": "robot", "episode_index": 0, "failure_reason": "out_of_bounds"}
    (tmp_path / "record.json").write_text(json.dumps(rec))
    rc = repro.load_failures(tmp_path / "record.json")
    assert rc[0].source == "record" and rc[0].failure_reason == "out_of_bounds"


def test_baseline_twin_misses_and_knob_search_finds_hidden_physics(tmp_path):
    _record_robot_failures(tmp_path / "robot")
    cases = repro.load_failures(tmp_path / "robot", domain="real")[:3]
    cfg = repro.load_config(None, env="reach2d", policy="pd", repeats=2, seed_jitter=3,
                            knobs={"actuator_gain": [1.0, 3.0, 4.0], "action_delay": [0, 5, 7]})
    summary = repro.reproduce_batch(cases, cfg)
    assert summary["reproduction_rate"] < 1.0  # the nominal twin does not fail like the robot
    assert summary["reproduction_rate_with_knobs"] == 1.0
    for r in summary["results"]:
        assert r["replay"]["nrmse"] > 0.1  # same commands, different motion
        assert r["verdict"] == "reproduced_with_knobs"
        assert r["twin_ticket"]["knobs"] == HIDDEN  # the replay pins the hidden physics exactly
    assert summary["twin_tickets"]["actuator_gain=3.0"] == 3
    md = repro.repro_markdown(summary, cfg)
    assert "Twin tickets" in md and "Failure reproduction rate" in md
    json.loads(repro.to_json(summary))


def test_reproduced_when_twin_matches_and_flags_missing_knobs(tmp_path):
    _record_robot_failures(tmp_path / "robot")
    cases = repro.load_failures(tmp_path / "robot")[:1]
    good = repro.load_config(None, env="reach2d", policy="pd", repeats=2, seed_jitter=0)
    case = cases[0]
    case.scenario = Scenario(seed=case.scenario.seed, params=HIDDEN, task="reach")
    r = repro.reproduce(case, good)
    assert r["verdict"] == "reproduced" and r["replay"]["nrmse"] < 1e-6
    bare = repro.load_config(None, env="reach2d", policy="pd", repeats=1, seed_jitter=0)
    r2 = repro.reproduce(repro.load_failures(tmp_path / "robot")[0], bare)
    assert r2["verdict"] != "reproduced"
    assert "suggested for this failure mode" in r2["search"]["notes"][0]


def test_cli_repro_writes_reports(tmp_path):
    from groot_eval.twin_cli import main

    _record_robot_failures(tmp_path / "robot", seeds=range(3))
    code = main(["repro", str(tmp_path / "robot"), "--env", "reach2d", "--policy", "pd", "--repeats", "1",
                 "--seed-jitter", "1", "--knob", "actuator_gain=1.0,3.0", "--knob", "action_delay=0,5",
                 "--out", str(tmp_path / "out")])
    assert code == 0
    assert (tmp_path / "out" / "repro.md").exists() and (tmp_path / "out" / "repro.json").exists()
