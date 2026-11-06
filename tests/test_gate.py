import json

import numpy as np
import pytest

from groot_eval.cli import main
from groot_eval.closed_loop import evaluate, run_episode
from groot_eval.envs.reach2d import Reach2DEnv
from groot_eval.latency import LatencyModel, RealTimePolicy
from groot_eval.manifest import build_manifest, diff_manifests, sha256_path, unpinned
from groot_eval.policies import make_policy
from groot_eval.randomization import SweepConfig
from groot_eval.regression import export_suite, find_flaky, parse_tolerances, run_suite
from groot_eval.types import Scenario

CFG = "seed: 4\nepisodes: 30\nparams:\n  obstacle: {choice: [false, true]}\n  friction: {uniform: [0.4, 1.0]}\n"


def _suite(tmp_path, policy="pd"):
    run = evaluate(Reach2DEnv(), make_policy(policy), SweepConfig.from_dict(
        {"seed": 4, "episodes": 30, "params": {"obstacle": {"choice": [False, True]}}}).scenarios(), env_name="reach2d")
    return export_suite(run, tmp_path / "suite.json", anchors=4)


class EveryOtherEpisode:
    """Deterministic per call but not per scenario: succeeds on even episodes, stands still on odd ones."""

    name = "flip"

    def __init__(self):
        self.inner, self.episode = make_policy("pd-avoid"), -1

    def reset(self, task):
        self.episode += 1
        self.inner.reset(task)

    def act(self, obs):
        return self.inner.act(obs) if self.episode % 2 == 0 else np.zeros(2)


# -- manifests ----------------------------------------------------------------------
def test_checkpoint_hash_tracks_content(tmp_path):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "model.safetensors").write_bytes(b"weights-v1")
    (ckpt / "config.json").write_text("{}", encoding="utf-8")
    h1 = sha256_path(ckpt)
    assert h1 == sha256_path(ckpt)
    (ckpt / "model.safetensors").write_bytes(b"weights-v2")
    assert sha256_path(ckpt) != h1


def test_manifest_diff_and_unpinned(tmp_path):
    cfg = tmp_path / "a.yaml"
    cfg.write_text(CFG, encoding="utf-8")
    a = build_manifest(policy="pd", env="reach2d", config=cfg)
    assert a["config_sha256"] and a["packages"]["numpy"]
    assert diff_manifests(a, json.loads(json.dumps(a))) == []
    b = {**a, "env_args": {"max_steps": 100}, "policy": "pd-avoid"}
    assert diff_manifests(a, b, ignore=("policy",)) == ['env_args: {} -> {"max_steps": 100}']
    assert diff_manifests(a, None)
    groot = build_manifest(policy="groot", env="reach2d", latency="measured (hold)", extra={"policy_spec": "groot://gpu:5555"})
    reasons = " ".join(unpinned({**groot, "policy": "groot://gpu:5555"}))
    assert "--checkpoint" in reasons and "measured" in reasons


def test_sweep_writes_manifest_and_suite_metrics(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(CFG, encoding="utf-8")
    out = tmp_path / "base"
    ckpt = tmp_path / "w.bin"
    ckpt.write_bytes(b"x")
    assert main(["sweep", "--policy", "pd", "--config", str(cfg), "--out", str(out), "--export-suite",
                 "--checkpoint", str(ckpt)]) == 0
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["checkpoint"]["sha256"] == sha256_path(ckpt) and m["config_sha256"] == sha256_path(cfg)
    suite = json.loads((out / "suite.json").read_text(encoding="utf-8"))
    assert suite["manifest"]["config_sha256"] == m["config_sha256"]
    assert all("final_distance" in c["metrics"] for c in suite["cases"])


def test_compare_strict_refuses_different_conditions(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(CFG, encoding="utf-8")
    other = tmp_path / "other.yaml"
    other.write_text(CFG.replace("seed: 4", "seed: 5"), encoding="utf-8")
    assert main(["sweep", "--policy", "pd", "--config", str(cfg), "--out", str(tmp_path / "a")]) == 0
    assert main(["sweep", "--policy", "pd-avoid", "--config", str(cfg), "--out", str(tmp_path / "b")]) == 0
    assert main(["sweep", "--policy", "pd-avoid", "--config", str(other), "--out", str(tmp_path / "c")]) == 0
    assert main(["compare", str(tmp_path / "a"), str(tmp_path / "b"), "--strict"]) == 0  # only the checkpoint differs
    assert main(["compare", str(tmp_path / "a"), str(tmp_path / "c"), "--strict"]) == 2


# -- repeats, flakes, quarantine, margins -----------------------------------------------
def test_flaky_cases_are_reported_not_gated(tmp_path):
    suite = _suite(tmp_path)
    r = run_suite(Reach2DEnv(), EveryOtherEpisode(), suite, repeats=2)
    assert len(r["flaky"]) == r["cases"] and r["passed_gate"]
    assert all(e["pass_rate"] == 0.5 for e in r["flaky"])
    single = run_suite(Reach2DEnv(), EveryOtherEpisode(), suite, repeats=1)
    assert single["cases"] == len(single["fixed"]) + len(single["still_failing"]) + len(single["broken_anchors"]) + single["anchors_ok"]


def test_quarantine_and_find_flaky(tmp_path):
    suite = _suite(tmp_path)
    flaky = find_flaky(Reach2DEnv(), EveryOtherEpisode(), suite, repeats=2)
    assert len(flaky) == len(suite["cases"])
    assert find_flaky(Reach2DEnv(), make_policy("pd"), suite, repeats=3) == []  # deterministic sim + policy
    r = run_suite(Reach2DEnv(), EveryOtherEpisode(), suite, repeats=2, quarantine={c["scenario_id"] for c in flaky})
    assert len(r["quarantined"]) == r["cases"] and not r["flaky"]


def test_metric_margin_fails_a_passing_anchor(tmp_path):
    suite = _suite(tmp_path)
    anchor = next(c for c in suite["cases"] if c["origin"] == "anchor")
    anchor["metrics"]["path_length"] = anchor["metrics"]["path_length"] - 0.3  # baseline was "better"
    r = run_suite(Reach2DEnv(), make_policy("pd"), suite, tolerances=parse_tolerances(["path_length:lower:0.1"]))
    assert not r["passed_gate"] and [d["over_tolerance"] for d in r["degraded_anchors"]] == [["path_length"]]
    ok = run_suite(Reach2DEnv(), make_policy("pd"), suite, tolerances=parse_tolerances(["path_length:lower:0.5"]))
    assert ok["passed_gate"]
    for bad in ("path_length", "path_length:down:0.1", "path_length:lower:-1"):
        with pytest.raises(ValueError):
            parse_tolerances([bad])


# -- latency determinism --------------------------------------------------------------------
def test_sampled_latency_depends_on_the_scenario_not_the_run_order():
    scen = [Scenario(seed=s, params={"obstacle": True}) for s in (3, 8, 13)]

    def steps(order):
        pol = RealTimePolicy(make_policy("pd-chunk", avoid=2.0, execute_steps=4), LatencyModel.parse("normal:110,40"),
                             0.05, "continue")
        return {s.seed: (r.success, r.steps) for s in order for r in [run_episode(Reach2DEnv(), pol, s)]}

    assert steps(scen) == steps(list(reversed(scen)))


def test_cli_gate_flow(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(CFG, encoding="utf-8")
    base = tmp_path / "base"
    assert main(["sweep", "--policy", "pd", "--config", str(cfg), "--out", str(base), "--export-suite"]) == 0
    suite = str(base / "suite.json")
    assert main(["regress", "--policy", "pd", "--suite", suite, "--latency-mode", "measured"]) == 2
    q = tmp_path / "quarantine.json"
    assert main(["flaky", "--policy", "pd", "--suite", suite, "--out", str(q)]) == 0
    data = json.loads(q.read_text(encoding="utf-8"))
    assert data["repeats"] == 5 and data["flaky"] == []
    out = tmp_path / "regress.md"
    assert main(["regress", "--policy", "pd-avoid", "--suite", suite, "--repeats", "2", "--quarantine", str(q),
                 "--metric-tolerance", "final_distance:lower:0.05", "--out", str(out), "--strict"]) == 0
    text = out.read_text(encoding="utf-8")
    assert "2 runs each" in text and "## Flaky" in text
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["manifest"]["policy_spec"] == "pd-avoid"
    assert main(["regress", "--policy", "pd-avoid", "--suite", suite, "--env-arg", "max_steps=150", "--strict"]) == 2
