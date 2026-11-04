from groot_eval.analysis import attribute_failures, worst_cases
from groot_eval.closed_loop import evaluate
from groot_eval.envs.reach2d import Reach2DEnv
from groot_eval.policies import make_policy
from groot_eval.randomization import SweepConfig
from groot_eval.regression import check_determinism, export_suite, load_suite, run_suite

CFG = SweepConfig.from_dict({
    "seed": 1, "episodes": 80,
    "params": {"obstacle": {"choice": [False, True]}, "friction": {"uniform": [0.3, 1.0]}},
})


def _run(spec):
    return evaluate(Reach2DEnv(), make_policy(spec), CFG.scenarios(), env_name="reach2d")


def test_attribution_finds_the_obstacle():
    findings = attribute_failures(_run("pd").results)
    top = findings[0]
    assert top["param"] == "obstacle"
    assert top["significant"]
    assert top["worst_bucket"] == "True"


def test_suite_round_trip(tmp_path):
    base = _run("pd")
    path = tmp_path / "suite.json"
    export_suite(base, path, anchors=3)
    suite = load_suite(path)
    failures = sum(not r.success for r in base.results)
    assert len(suite["cases"]) == failures + 3

    same = run_suite(Reach2DEnv(), make_policy("pd"), suite)
    assert not same["fixed"] and len(same["still_failing"]) == failures and same["passed_gate"]

    better = run_suite(Reach2DEnv(), make_policy("pd-avoid"), suite)
    assert len(better["fixed"]) > failures // 2


def test_simulator_is_deterministic_per_scenario():
    scenarios = list(CFG.scenarios())[:10]
    assert check_determinism(Reach2DEnv(), make_policy("pd"), scenarios) == []


def test_worst_cases_only_lists_failures():
    worst = worst_cases(_run("pd").results, limit=5)
    assert worst and all(not w["success"] for w in worst)
