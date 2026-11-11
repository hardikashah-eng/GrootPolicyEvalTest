"""Walker twin on a treadmill: belts, recentering, perturbations, catalogue, belt replay and the CLI."""

import json

import numpy as np
import pytest

from groot_eval.closed_loop import evaluate, run_episode
from groot_eval.envs import make_env
from groot_eval.envs.treadmill import (Perturbation, PerturbationConfig, Treadmill, perturbed_scenarios,
                                       schedule_label, trip_offset)
from groot_eval.policies import make_policy
from groot_eval.replay import replay_trace
from groot_eval.robot_cli import main
from groot_eval.trace import TraceRecorder, TraceSet
from groot_eval.types import Scenario


def _walk(params=None, policy="walker-capture", env="treadmill", seed=0):
    return run_episode(make_env(env), make_policy(policy), Scenario(seed=seed, params=params or {}))


def _trip(sev, side="left"):
    return {"treadmill.perturbations": [{"type": "trip", "side": side, "start_s": 5, "severity": sev}]}


def test_overground_walking_matches_the_commanded_speed():
    r = _walk({}, env="walker")
    assert r.success and r.metrics["steps"] == 50
    assert r.metrics["step_len_left"] == pytest.approx(0.6 * 0.4, abs=0.01)  # speed x step time


def test_treadmill_controller_stays_on_the_deck_and_the_blind_one_needs_recentering():
    for speed in (0.8, 1.2):
        r = _walk({"treadmill.speed": speed})
        assert r.success and r.metrics["deck_excursion"] < 0.05
        assert r.metrics["step_len_left"] == pytest.approx(speed * 0.4, abs=0.02)
    blind = _walk({}, policy="walker-blind")
    assert not blind.success and blind.failure_reason == "off_belt"
    assert _walk({"treadmill.mode": "recenter"}, policy="walker-blind").success


def test_split_belt_makes_the_gait_asymmetric():
    even, split = _walk({}), _walk({"treadmill.split": 1.3})
    assert split.success and split.metrics["step_asymmetry"] > 5 * even.metrics["step_asymmetry"]
    assert split.metrics["step_len_left"] > split.metrics["step_len_right"]  # the right belt is faster


def test_trip_profile_and_severity_threshold():
    assert trip_offset(0.05, 10) == pytest.approx(-0.5)
    assert trip_offset(0.32, 10) == pytest.approx(2.2)  # surge peak
    assert trip_offset(0.54, 10) == pytest.approx(0.0, abs=1e-9)
    assert _walk(_trip(2)).success and _walk(_trip(4)).success
    severe = _walk(_trip(14))
    assert not severe.success and severe.failure_reason == "fall"
    assert severe.metrics["perturbations_fired"] == 1


def test_trip_fires_at_touchdown_of_its_side_and_is_repeatable():
    env = make_env("treadmill")
    run_episode(env, make_policy("walker-capture"), Scenario(seed=3, params=_trip(3, "right")))
    log = env.treadmill.log
    assert len(log) == 1 and log[0]["t"] >= 5.0 and log[0]["step"] % 2 == 1  # right foot lands on odd steps
    a, b = _walk(_trip(9), seed=7), _walk(_trip(9), seed=7)
    assert (a.success, a.steps, a.metrics) == (b.success, b.steps, b.metrics)


def test_slip_push_incline_and_speed_step():
    assert _walk({"treadmill.perturbations": [{"type": "slip", "side": "right", "start_s": 5, "severity": 0.05,
                                               "duration_s": 0.5}]}).metrics["slip_distance"] > 0
    assert not _walk({"treadmill.perturbations": [{"type": "push", "start_s": 6, "severity": 300,
                                                   "duration_s": 0.2}]}).success
    assert _walk({"treadmill.incline_deg": 6}).metrics["deck_excursion"] > 0.2  # drifts back on a slope
    assert _walk({"treadmill.perturbations": [{"type": "speed_step", "start_s": 5, "severity": 0.4,
                                               "duration_s": 3}]}).success


def test_belt_speed_is_recorded_and_replayed(tmp_path):
    rec = TraceRecorder(domain="real", env="treadmill", policy="walker-blind", control_dt=0.02)
    evaluate(make_env("treadmill"), make_policy("walker-blind"),
             [Scenario(seed=i, params={"treadmill.mode": "recenter"}) for i in range(2)], recorder=rec)
    root = rec.write(tmp_path / "real")
    ts = TraceSet.open(root)
    assert "belt_speed" in ts.state_slices()
    belt = ts.per_episode("observation.state")[0][:, ts.state_slices()["belt_speed"]]
    assert belt.std() > 0.01  # the recentering loop moved the belt
    env = make_env("treadmill")
    nominal = replay_trace(env, ts, {"treadmill.speed": 0.8}, reset_every=10)
    recorded = replay_trace(env, ts, {"treadmill.belt": f"trace:{root}"}, reset_every=10)
    assert recorded.score < 0.01 < nominal.score
    with pytest.raises(KeyError, match="seed"):
        Treadmill({"belt": f"trace:{root}"}, seed=99, dt=0.02)


def test_perturbation_catalogue_is_seeded_per_scenario():
    cfg = PerturbationConfig.load("configs/treadmill/single_perturbation.yaml")
    base = [Scenario(seed=i) for i in range(40)]
    a, b = perturbed_scenarios(base, cfg, seed=5), perturbed_scenarios(base, cfg, seed=5)
    assert [s.params for s in a] == [s.params for s in b]
    assert [s.params for s in a] != [s.params for s in perturbed_scenarios(base, cfg, seed=6)]
    # adding scenarios does not change the schedules of the ones already there
    assert [s.params for s in perturbed_scenarios(base[:10], cfg, seed=5)] == [s.params for s in a[:10]]
    kinds = {schedule_label(s.params) for s in a}
    assert kinds <= {"trip", "slip", "speed_step", "split", "incline_step", "push"} and len(kinds) >= 4
    for s in a:
        p = s.params["treadmill.perturbations"][0]
        assert 3 <= p["start_s"] <= 10 and s.params["treadmill.speed"] == 0.8
    rnd = PerturbationConfig.from_dict({"perturbations": {"mode": "random", "rate_per_min": 30, "max_events": 3,
                                                          "catalogue": [{"type": "push", "severity": [10, 20]}]}})
    assert 1 <= len(rnd.sample(np.random.default_rng(0), 20.0)) <= 3


def test_validation():
    with pytest.raises(ValueError, match="unknown perturbation"):
        Perturbation(type="earthquake", severity=1)
    with pytest.raises(ValueError, match="unknown treadmill parameters"):
        Treadmill({"sped": 1.0}, seed=0, dt=0.02)
    with pytest.raises(ValueError, match="Walker does not know"):
        _walk({"wobble": 1})


def test_set_state_round_trip():
    env = make_env("treadmill")
    pol = make_policy("walker-capture")
    obs = env.reset(Scenario(seed=1))
    for _ in range(137):  # past the belt ramp-up, so the treadmill clock does not matter
        obs, *_ = env.step(pol.act(obs))
    snap = {k: v for k, v in obs.items() if k.startswith("state.")}
    a1, *_ = env.step(pol.act(obs))
    env.set_state(snap)
    a2, *_ = env.step(pol.act(obs))
    for k in ("state.com_pos", "state.com_vel", "state.com_rel", "state.belt_speed"):
        assert np.allclose(a1[k], a2[k], atol=1e-4), k


def test_cli_treadmill_and_replay(tmp_path):
    assert main(["treadmill", "--config", "configs/treadmill/trip_ladder.yaml", "--episodes", "16", "--seed", "3",
                 "--record", "--out", str(tmp_path / "t")]) == 0
    res = json.loads((tmp_path / "t" / "treadmill.json").read_text())
    trips = res["summary"][0]
    assert trips["perturbation"] == "trip" and trips["episodes"] == 16 and 0 < trips["kept_walking"] < 16
    again = tmp_path / "t2"
    assert main(["treadmill", "--config", "configs/treadmill/trip_ladder.yaml", "--episodes", "16", "--seed", "3",
                 "--out", str(again)]) == 0
    assert json.loads((again / "treadmill.json").read_text())["episodes"] == res["episodes"]  # same seed, same run
    assert main(["treadmill-replay", str(tmp_path / "t" / "trace"), "--episodes", "4", "--out", str(tmp_path / "r")]) == 0
    rep = json.loads((tmp_path / "r" / "treadmill_replay.json").read_text())
    assert rep["recorded"] < rep["nominal"]
