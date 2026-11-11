"""Time-to-success distributions, safety metrics and paired comparisons."""

import json
import math

import numpy as np

from groot_eval import distributional as ds


def eps(times, ok, ids=None):
    return [{"scenario_id": (ids or [str(i) for i in range(len(times))])[i], "steps": t, "success": o}
            for i, (t, o) in enumerate(zip(times, ok))]


def test_kaplan_meier_and_rmst_known_values():
    t, e = np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1])
    ut, s = ds.kaplan_meier(t, e)
    assert ut.tolist() == [1.0, 2.0, 3.0] and np.allclose(s, [2 / 3, 1 / 3, 0])
    assert math.isclose(ds.rmst(t, e, 3.0), 2.0)
    assert ds.median_time(t, e) == 2.0
    # a failure is censored, not a fast success: it keeps the curve up and costs the horizon
    t2, e2 = np.array([1.0, 2.0, 3.0]), np.array([1, 1, 0])
    assert math.isclose(ds.rmst(t, e, 4.0), 2.0) and math.isclose(ds.rmst(t2, e2, 4.0), 2.0 + 1 / 3)
    assert math.isinf(ds.median_time(np.array([5.0, 6.0]), np.array([0, 0])))


def test_time_to_success_summary():
    out = ds.time_to_success(eps([10, 20, 30, 40, 200], [1, 1, 1, 1, 0]), dt=0.05, horizon=10.0)
    assert out["episodes"] == 5 and out["successes"] == 4
    assert math.isclose(out["median_s"], 1.5)
    lo, hi = out["rmst_ci"]
    assert lo <= out["rmst_s"] <= hi


def test_paired_comparison_detects_slower_run():
    rng = np.random.default_rng(0)
    base = rng.integers(20, 60, 80)
    a = eps(base.tolist(), [1] * 80)
    b = eps((base + 15).tolist(), [1] * 80)  # same success, 15 steps slower on every scenario
    c = ds.compare(a, b, dt=0.05)
    assert c["pairs"] == 80 and c["joined_by"] == "scenario_id"
    assert c["rmst_diff_s"] > 0.5 and c["rmst_diff_ci"][0] > 0
    assert c["verdict"] == "B worse"
    assert ds.compare(a, a, dt=0.05)["verdict"] == "no significant difference"


def test_paired_comparison_success_flip():
    a = eps([30] * 40, [1] * 40)
    b = eps([30] * 40, [1] * 25 + [0] * 15)
    c = ds.compare(a, b, dt=1.0)
    assert c["only_a_succeeded"] == 15 and c["only_b_succeeded"] == 0 and c["mcnemar_p"] < 0.01
    assert c["verdict"] == "B worse"


def test_episode_safety():
    t = np.linspace(0, 1, 50)[:, None]
    smooth = np.c_[np.sin(t), np.cos(t)]
    jumpy = smooth.copy()
    jumpy[25] += 0.8
    s, j = ds.episode_safety(smooth, 0.02), ds.episode_safety(jumpy, 0.02)
    assert j["max_jump"] > 0.7 > s["max_jump"] and j["jerk_rms"] > 10 * s["jerk_rms"]
    states = np.c_[np.linspace(-0.99, 0.0, 50)]
    near = ds.episode_safety(smooth, 0.02, states, (np.array([-1.0]), np.array([1.0])))
    assert 0 < near["near_limit_share"] < 0.2
    sat = ds.episode_safety(np.clip(smooth * 3, -1, 1), 0.02, action_range=(-1.0, 1.0))
    assert sat["saturation_share"] > 0.3
    summary = ds.safety_summary([s, j], {"max_jump": 0.5})
    assert summary["metrics"]["max_jump"]["over"] == 1


def test_safety_from_trace_and_cli(tmp_path):
    from groot_eval.cli import main
    from groot_eval.envs import make_env
    from groot_eval.policies import make_policy
    from groot_eval.trace import TraceRecorder
    from groot_eval.types import Scenario

    env, pol = make_env("reach2d"), make_policy("pd")
    rec = TraceRecorder(domain="sim", env="reach2d", policy="pd", control_dt=env.dt)
    runs = {"a": [], "b": []}
    for seed in range(12):
        for name, delay in (("a", 0), ("b", 3)):
            sc = Scenario(seed=seed, params={"action_delay": delay})
            obs = env.reset(sc)
            if name == "a":
                rec.start_episode(env.task, scenario=sc.to_dict())
            info, steps = {}, 0
            for _ in range(200):
                steps += 1
                act = pol.act(obs)
                if name == "a":
                    rec.record_step(obs, act)
                obs, _, term, trunc, info = env.step(act)
                if term or trunc:
                    break
            if name == "a":
                rec.end_episode(success=bool(info.get("success")))
            runs[name].append({"scenario_id": f"s{seed}", "steps": steps, "success": bool(info.get("success"))})
    root = rec.write(tmp_path / "trace_a")
    rows = ds.safety_from_trace(root, action_range=(-1.0, 1.0))
    assert len(rows) == 12 and all("jerk_rms" in r and "saturation_share" in r for r in rows)
    for name, items in runs.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in items) + "\n")
    out = tmp_path / "out"
    code = main(["dist", str(tmp_path / "a"), "--vs", str(tmp_path / "b"), "--dt", "0.05", "--trace", str(root),
                 "--action-range=-1,1", "--out", str(out)])
    assert code == 0
    payload = json.loads((out / "dist.json").read_text())
    assert payload["comparison"]["pairs"] == 12 and payload["safety"]["A"]["episodes"] == 12
    assert "Time to success and safety" in (out / "dist.md").read_text()
    # a strict jump limit turns the gate red
    assert main(["dist", str(tmp_path / "a"), "--trace", str(root), "--threshold", "max_jump=0.0001", "--gate",
                 "--out", str(tmp_path / "o2")]) == 1
