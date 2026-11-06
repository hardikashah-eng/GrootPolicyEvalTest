import numpy as np
import pandas as pd
import pytest

from groot_eval.cli import main
from groot_eval.correlation import correlate, join_diagnostics, load_pairs, pairs_from_trace
from groot_eval.randomization import SweepConfig
from groot_eval.trace import TraceRecorder, TraceSet
from groot_eval.types import Scenario

NAMED = """\
task: reach
repeats: 2
params:
  friction: {fixed: 0.7}
scenarios:
  - {name: pick_place_00, seed: 0, params: {obstacle: true}}
  - {name: pick_place_01, seed: 10}
  - {name: pick_place_02, seed: 20, params: {friction: 1.1}}
"""


def test_named_scenarios_in_a_sweep_config(tmp_path):
    cfg = tmp_path / "named.yaml"
    cfg.write_text(NAMED, encoding="utf-8")
    scen = list(SweepConfig.load(cfg).scenarios())
    assert [s.name for s in scen] == ["pick_place_00"] * 2 + ["pick_place_01"] * 2 + ["pick_place_02"] * 2
    assert [s.seed for s in scen[:2]] == [0, 1]
    assert scen[0].params == {"friction": 0.7, "obstacle": True} and scen[4].params == {"friction": 1.1}
    assert scen[0].pair_id == "pick_place_00"


@pytest.mark.parametrize("bad, match", [
    ({"scenarios": [{"seed": 1}]}, "needs a name"),
    ({"scenarios": [{"name": "a"}, {"name": "a"}]}, "unique"),
    ({"params": {"mass": {"uniform": [1, 2]}}, "scenarios": [{"name": "a"}]}, "only be fixed"),
])
def test_named_scenario_validation(bad, match):
    with pytest.raises(ValueError, match=match):
        SweepConfig.from_dict(bad)


def test_name_does_not_change_the_id_and_round_trips():
    plain, named = Scenario(seed=3, params={"mass": 1.0}), Scenario(seed=3, params={"mass": 1.0}, name="cup_07")
    assert plain.scenario_id == named.scenario_id and plain.pair_id == plain.scenario_id
    assert Scenario.from_dict(named.to_dict()) == named
    assert Scenario.from_dict({"seed": 1}).name == ""  # old suites and results without names still load


def _real_trace(tmp_path, outcomes):
    rec = TraceRecorder(domain="real", policy="ckpt_a", control_dt=0.05)
    for name, ok in outcomes.items():
        rec.start_episode("reach", scenario_name=name)
        for t in range(5):
            rec.record_step({"state.q": np.array([t * 0.1])}, np.array([0.0]))
        rec.end_episode(success=ok)
    return rec.write(tmp_path / "real")


def test_sweep_and_real_trace_join_in_correlate(tmp_path):
    cfg = tmp_path / "named.yaml"
    cfg.write_text(NAMED, encoding="utf-8")
    pairs = tmp_path / "pairs.csv"
    for policy, label in (("pd", "ckpt_a"), ("pd-avoid", "ckpt_b")):
        assert main(["sweep", "--policy", policy, "--config", str(cfg), "--out", str(tmp_path / label),
                     "--export-pairs", str(pairs), "--pairs-policy", label]) == 0
    real = _real_trace(tmp_path, {"pick_place_00": False, "pick_place_01": True, "pick_place_02": True})
    assert main(["pairs", str(real), "--out", str(pairs)]) == 0
    assert main(["pairs", str(real), "--out", str(pairs), "--policy", "ckpt_b"]) == 0
    df = load_pairs(pairs)
    assert len(df) == 2 * 6 + 2 * 3 and set(df["scenario_id"]) == {"pick_place_00", "pick_place_01", "pick_place_02"}
    c = correlate(df)
    assert c["matched_pairs"] == 6 and c["join"]["warnings"] == []
    out = tmp_path / "corr.md"
    assert main(["correlate", str(pairs), "--out", str(out)]) == 0
    assert "6 policy/scenario pairs matched" in out.read_text(encoding="utf-8")


def test_hash_ids_against_names_are_explained(make_trace):
    sim = pairs_from_trace(make_trace("sim", episodes=3), policy="ckpt_a")
    assert all(len(r["scenario_id"]) == 12 for r in sim)  # unnamed sim scenarios fall back to the hash id
    real = [{"policy": "ckpt_a", "scenario_id": f"pick_place_{i:02d}", "domain": "real", "success": 1} for i in range(3)]
    j = join_diagnostics(pd.DataFrame(sim + real))
    assert j["matched"] == 0 and any("generated hashes" in w for w in j["warnings"])
    other = [{**r, "policy": "robot_v2"} for r in real]
    sim_named = [{**r, "domain": "sim"} for r in real]
    assert any("no policy appears in both" in w for w in join_diagnostics(pd.DataFrame(sim_named + other))["warnings"])


def test_partial_overlap_is_reported():
    rows = [{"policy": "a", "scenario_id": s, "domain": d, "success": 1}
            for s, d in (("s1", "sim"), ("s1", "real"), ("s2", "sim"), ("s3", "real"))]
    j = join_diagnostics(pd.DataFrame(rows))
    assert j["matched"] == 1 and j["only_sim"] == ["s2"] and j["only_real"] == ["s3"]
    assert any("left out" in w for w in j["warnings"])


def test_pairs_from_trace_needs_success_and_domain(tmp_path):
    rec = TraceRecorder(domain="real", control_dt=0.05)
    rec.start_episode("t")
    rec.record_step({"state.q": np.zeros(1)}, np.zeros(1))
    rec.end_episode()  # no success mark
    ts = TraceSet.open(rec.write(tmp_path / "t"))
    assert pairs_from_trace(ts) == []
    assert main(["pairs", str(ts.root), "--out", str(tmp_path / "p.csv")]) == 1
