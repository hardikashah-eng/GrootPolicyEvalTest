import math

import numpy as np

from groot_eval import twin_metrics as tm


def test_mmrv_zero_for_same_ranking_and_weighted_by_real_gap():
    assert tm.mmrv([0.9, 0.7, 0.5], [0.8, 0.6, 0.2]) == 0.0
    # sim swaps two policies that are 20 points apart on the robot
    assert math.isclose(tm.mmrv([0.7, 0.9], [0.9, 0.7]), 0.2)


def test_kendall_and_pearson():
    assert tm.kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert tm.kendall_tau([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
    assert math.isnan(tm.pearson([1, 1, 1], [1, 2, 3]))


def test_calibration_recovers_bias_and_slope():
    sim = np.array([0.2, 0.4, 0.6, 0.8])
    cal = tm.calibration(sim, sim - 0.1)
    assert math.isclose(cal["bias"], 0.1, abs_tol=1e-9)
    assert math.isclose(cal["slope"], 1.0, abs_tol=1e-9)


def test_predictive_validity_and_sensitivity_from_rows():
    rows = []
    real = {"a": 0.9, "b": 0.6, "c": 0.3, "d": 0.5, "e": 0.8}
    for p, r in real.items():
        for dom, rate in (("real", r), ("sim", min(1.0, r + 0.05))):
            rows.append({"policy": p, "domain": dom, "condition": "nominal", "successes": int(rate * 40), "trials": 40})
            # lighting hurts the real camera policy, the twin does not model it
            drop = 0.3 if dom == "real" else 0.0
            rows.append({"policy": p, "domain": dom, "condition": "dim_light",
                         "successes": int(max(rate - drop, 0) * 40), "trials": 40})
    pv = tm.predictive_validity(rows, gate=0.55)
    assert pv["policies"] == 5
    assert pv["pearson"] > 0.95 and pv["mmrv"] == 0.0
    assert pv["calibration"]["bias"] > 0
    sens = pv["sensitivity"]
    assert sens["pairs"] == 5 and sens["mean_abs_error"] > 0.2
    assert next(iter(sens["by_condition"])) == "dim_light"
    assert pv["decisions"]["decisions"] == 5


def test_decision_agreement_counts():
    d = tm.decision_agreement([True, False, True, False], [True, True, False, False])
    assert (d["tp"], d["fp"], d["fn"], d["tn"]) == (1, 1, 1, 1)
    assert d["recall"] == 0.5


def test_trajectory_and_actuator_metrics():
    t = np.linspace(0, 10, 300)
    rec = np.stack([np.sin(t), np.cos(t)], 1)
    assert np.allclose(tm.nrmse(rec, rec), 0)
    assert tm.divergence_step(rec, rec) is None
    late = np.vstack([rec[:1].repeat(5, 0), rec[:-5]])
    assert tm.dtw_distance(late, rec) < float(np.mean(tm.nrmse(late, rec)))  # timing, not shape
    cmd = np.sin(t)[:, None]
    meas = np.vstack([np.zeros((3, 1)), 0.8 * cmd[:-3]])
    row = tm.actuator_tracking(cmd, meas)[0]
    assert row["lag_steps"] == 3 and abs(row["gain"] - 0.8) < 0.02


def test_latency_and_failure_modes():
    lm = tm.latency_match([10] * 50, [110] * 50, control_dt_ms=50)
    assert math.isclose(lm["w1_ms"], 100) and math.isclose(lm["w1_steps"], 2)
    conf = tm.failure_mode_confusion([("collision", "collision"), ("collision", "success"), ("timeout", "timeout")])
    assert math.isclose(conf["agreement"], 2 / 3)
    assert conf["per_mode"]["collision"]["recall"] == 0.5
    rs = tm.reproduction_score("collision", [(False, "collision"), (False, "timeout"), (True, None), (False, "collision")])
    assert rs["same_mode_rate"] == 0.5 and rs["fail_rate"] == 0.75


def test_visual_metrics():
    rng = np.random.default_rng(0)
    img = (rng.random((40, 48, 3)) * 255).astype(np.uint8)
    assert tm.psnr(img, img) == math.inf and math.isclose(tm.ssim(img, img), 1.0)
    assert tm.channel_order_check(img, img[..., ::-1])["swapped"]
    assert not tm.channel_order_check(img, img)["swapped"]
    frames = [np.full((8, 8), 10.0 * i) for i in range(20)]
    assert tm.best_frame_offset(frames, frames[1:])["offset"] == 1
    blurred = ((img[:-1, :-1].astype(float) + img[1:, 1:]) / 2).astype(np.uint8)
    assert tm.sharpness_ratio(img[:-1, :-1], blurred) < 0.8
    gap = tm.policy_visual_gap([img], [img], encoder=lambda f: f.mean(axis=(0, 1)), action_fn=lambda f: f.mean())
    assert gap["embedding_cosine_distance_mean"] < 1e-9 and gap["action_divergence_mean"] == 0


def test_value_metrics():
    cv = tm.control_variate_savings([1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0])
    assert 0 < cv["rho"] < 1 and cv["real_runs_multiplier"] > 1
    cov = tm.coverage_vs_field(np.random.default_rng(0).uniform(0, 1, (500, 2)), np.array([[0.5, 0.5], [3.0, 0.5]]))
    assert cov["coverage"] == 0.5
