import json

import numpy as np
import pytest

from groot_eval.cli import main
from groot_eval.gap import gap_analysis, gap_markdown, ks_pvalue, ks_statistic, wasserstein1
from groot_eval.trace import TraceRecorder, TraceSet


def test_distance_helpers():
    rng = np.random.default_rng(0)
    a = rng.normal(size=4000)
    assert wasserstein1(a, a + 0.5) == pytest.approx(0.5, abs=1e-6)
    assert ks_statistic(a, a) == 0 and ks_pvalue(0.0, 10, 10) == 1.0
    assert ks_pvalue(ks_statistic(a, a + 2), 20, 20) < 1e-3


def test_identical_traces_have_no_gap(make_trace):
    a, b = make_trace("a"), make_trace("b")
    r = gap_analysis(a, b)
    assert not [c for c in r["contract"] if c["level"] == "error"]
    assert r["signals"] and all(s["w1_norm"] == pytest.approx(0, abs=1e-9) for s in r["signals"])


def test_sensor_noise_shows_up_as_jerky_actions(make_trace):
    sim = make_trace("sim")
    real = make_trace("real", params={"sensor_noise": 0.05}, domain="real")
    r = gap_analysis(sim, real)
    changes = [s for s in r["signals"] if s["group"] == "action_change"]
    assert changes and all(s["w1_norm"] > 0.5 and s["b"]["mean"] > s["a"]["mean"] for s in changes)
    assert "jerkier in real" in changes[0]["hint"]
    top_groups = {s["group"] for s in r["signals"][:4]}
    assert "action_change" in top_groups


def _latency_trace(root, latency_ms):
    rec = TraceRecorder(domain="real", control_dt=0.05)
    for _ in range(3):
        rec.start_episode("t")
        for t in range(20):
            rec.record_step({"state.q": np.array([np.sin(t / 3)])}, np.array([0.1 * t]), latency_ms=latency_ms)
        rec.end_episode(success=True)
    return TraceSet.open(rec.write(root))


def test_unmodelled_latency_is_called_out(tmp_path):
    r = gap_analysis(_latency_trace(tmp_path / "s", 2.0), _latency_trace(tmp_path / "r", 80.0))
    assert r["timing"]["real"]["latency_p95_steps"] == pytest.approx(1.6)
    assert any("add about 2 step(s)" in n for n in r["timing_notes"])
    assert "## 3. Timing" in gap_markdown(r)


def test_cli_gap_report(make_trace, tmp_path):
    sim = make_trace("sim", policy="pd-chunk", execute_steps=4)
    real = make_trace("real", params={"action_delay": 2, "friction": 1.0}, domain="real", policy="pd-chunk", execute_steps=4)
    out = tmp_path / "gap"
    assert main(["gap", str(sim.root), str(real.root), "--out", str(out)]) == 0
    text = (out / "gap.md").read_text(encoding="utf-8")
    for heading in ("## 1. Contract", "## 2. Largest gaps", "## 4. Action chunking", "## 5. Outcomes"):
        assert heading in text
    data = json.loads((out / "gap.json").read_text(encoding="utf-8"))
    assert data["chunking"]["sim"]["boundaries"] > 0
    assert data["outcomes"]["real"]["episodes"] == 6
