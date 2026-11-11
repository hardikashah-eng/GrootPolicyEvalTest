"""Twin credibility per scenario class and the twin certificate."""

import csv
import datetime as dt
import json

import numpy as np

from groot_eval.credibility import assess, check_certificate, make_certificate

REAL = [0.20, 0.35, 0.50, 0.65, 0.80, 0.90]


def rows_for(cls, sim_rates, real_rates, n=40, seed=0, paired=True):
    rng = np.random.default_rng(seed)
    rows = []
    for i, (s, r) in enumerate(zip(sim_rates, real_rates)):
        u = rng.random(n)  # shared draw per scenario: sim and real agree where their rates overlap
        for k in range(n):
            sid = f"{cls}-{k}" if paired else ""
            rows.append({"policy": f"p{i}", "domain": "sim", "scenario_class": cls, "success": bool(u[k] < s), "scenario_id": sid})
            rows.append({"policy": f"p{i}", "domain": "real", "scenario_class": cls, "success": bool(u[k] < r), "scenario_id": sid})
    return rows


def evidence():
    good = rows_for("tabletop", [r + 0.02 for r in REAL], REAL)
    bad = rows_for("articulated", list(reversed(REAL)), REAL, seed=1)
    thin = rows_for("deformable", REAL[:3], REAL[:3], seed=2)
    return good + bad + thin


def test_verdict_per_class():
    res = assess(evidence(), gate=0.5)
    c = res["classes"]
    assert c["tabletop"]["verdict"] == "trusted", c["tabletop"]["failed"] + c["tabletop"]["thin_evidence"]
    assert c["tabletop"]["pearson"] > 0.95 and c["tabletop"]["mmrv"] < 0.05
    assert c["tabletop"]["paired"]["agreement"] > 0.9
    assert c["articulated"]["verdict"] == "not_trusted"
    assert any("MMRV" in f for f in c["articulated"]["failed"])
    assert c["deformable"]["verdict"] == "provisional"
    assert any("policies" in t for t in c["deformable"]["thin_evidence"])
    assert res["counts"] == {"trusted": 1, "provisional": 1, "not_trusted": 1}


def test_bias_alone_fails_a_class():
    res = assess(rows_for("optimistic", [min(1.0, r + 0.25) for r in REAL], REAL))
    m = res["classes"]["optimistic"]
    assert m["verdict"] == "not_trusted" and any("bias" in f for f in m["failed"])


def test_aggregated_rows_work():
    rows = [{"policy": f"p{i}", "domain": d, "scenario_class": "all", "successes": int(r * 50), "trials": 50}
            for i, r in enumerate(REAL) for d in ("sim", "real")]
    assert assess(rows)["classes"]["all"]["verdict"] == "trusted"


def test_certificate_rules():
    res = assess(evidence())
    cert = make_certificate(res, "arm3-twin", "1.4.0", {"name": "arm-07", "version": 3, "hash": "abc123"}, valid_days=30)
    assert check_certificate(cert, "tabletop", "abc123", "1.4.0")[0]
    ok, why = check_certificate(cert, "articulated")
    assert not ok and "not_trusted" in why
    assert not check_certificate(cert, "deformable")[0]
    assert check_certificate(cert, "deformable", allow_provisional=True)[0]
    assert not check_certificate(cert, "tabletop", profile_hash="zzz")[0]
    assert not check_certificate(cert, "tabletop", twin_version="1.5.0")[0]
    later = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=31)
    ok, why = check_certificate(cert, "tabletop", now=later)
    assert not ok and "expired" in why
    assert not check_certificate(cert, "unknown-class")[0]


def test_cli(tmp_path):
    from groot_eval.twin_cli import main

    rows = evidence()
    path = tmp_path / "results.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "out"
    assert main(["credibility", str(path), "--twin-version", "1.0", "--require", "tabletop", "--out", str(out)]) == 0
    assert main(["credibility", str(path), "--require", "articulated", "--out", str(tmp_path / "o2")]) == 1
    cert = out / "twin_certificate.json"
    assert json.loads(cert.read_text())["classes"]["tabletop"]["verdict"] == "trusted"
    assert "Twin credibility by scenario class" in (out / "credibility.md").read_text()
    assert main(["cert-check", str(cert), "--class", "tabletop"]) == 0
    assert main(["cert-check", str(cert), "--class", "articulated"]) == 1
