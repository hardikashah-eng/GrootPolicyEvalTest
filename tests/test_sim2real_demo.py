"""The CPU sim-to-real demo must keep finding what it hid (takes ~15 s)."""

import importlib.util
from pathlib import Path

import pytest

DEMO = Path(__file__).resolve().parent.parent / "examples" / "run_sim2real_demo.py"


def test_demo_recovers_the_hidden_robot(tmp_path):
    spec = importlib.util.spec_from_file_location("sim2real_demo", DEMO)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    r = demo.run(tmp_path / "out")
    assert r["fitted"] == pytest.approx(demo.HIDDEN)
    assert r["gap_calibrated"][0] < 0.5 * r["gap_naive"][0]
    assert r["corr_after"]["outcome_agreement"] > r["corr_before"]["outcome_agreement"]
    assert r["corr_after"]["join"]["warnings"] == []
    text = (tmp_path / "out" / "SUMMARY.md").read_text(encoding="utf-8")
    assert text.startswith("# Sim-to-real debugging demo (SYNTHETIC)")
    assert "units" in (tmp_path / "out" / "check_buggy.md").read_text(encoding="utf-8")
