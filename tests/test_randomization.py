import pytest

from groot_eval.randomization import SweepConfig, parse_param

CFG = {
    "seed": 3,
    "episodes": 50,
    "params": {
        "friction": {"uniform": [0.2, 1.0]},
        "delay": {"choice": [0, 2]},
        "obstacle": {"choice": [False, True], "weights": [1, 0]},
        "gravity": 9.81,
    },
}


def test_scenarios_are_reproducible_and_in_range():
    a = list(SweepConfig.from_dict(CFG).scenarios())
    b = list(SweepConfig.from_dict(CFG).scenarios())
    assert [s.scenario_id for s in a] == [s.scenario_id for s in b]
    assert len(a) == 50
    for s in a:
        assert 0.2 <= s.params["friction"] <= 1.0
        assert s.params["delay"] in (0, 2)
        assert s.params["obstacle"] is False  # zero weight never drawn
        assert s.params["gravity"] == 9.81
        assert isinstance(s.params["delay"], int)  # numpy scalars converted for JSON


def test_different_seed_gives_different_scenarios():
    other = dict(CFG, seed=4)
    a = {s.scenario_id for s in SweepConfig.from_dict(CFG).scenarios()}
    b = {s.scenario_id for s in SweepConfig.from_dict(other).scenarios()}
    assert not a & b


def test_grid_mode_covers_every_combination():
    cfg = SweepConfig.from_dict({"mode": "grid", "grid_points": 3, "repeats": 2,
                                 "params": {"friction": {"uniform": [0, 1]}, "delay": {"choice": [0, 1]}}})
    scenarios = list(cfg.scenarios())
    assert len(scenarios) == 3 * 2 * 2
    assert {s.params["friction"] for s in scenarios} == {0.0, 0.5, 1.0}


@pytest.mark.parametrize("spec", [{"uniform": [2, 1]}, {"loguniform": [0, 1]}, {"choice": []}, {"what": 1},
                                  {"choice": [1, 2], "weights": [1]}])
def test_bad_specs_rejected(spec):
    with pytest.raises(ValueError):
        parse_param("x", spec)
