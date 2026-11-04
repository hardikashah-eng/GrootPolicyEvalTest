import json

import pandas as pd
import pytest

from groot_eval.cli import main
from groot_eval.correlation import correlate, load_pairs


def test_perfect_agreement(tmp_path):
    rows = []
    for policy, wins in (("a", 2), ("b", 5), ("c", 8)):
        for i in range(10):
            for domain in ("sim", "real"):
                rows.append({"policy": policy, "scenario_id": f"s{i}", "domain": domain, "success": int(i < wins)})
    path = tmp_path / "pairs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    c = correlate(load_pairs(path))
    assert c["ranking_spearman"] == pytest.approx(1.0)
    assert c["outcome_agreement"] == 1.0
    assert all(p["gap"] == 0 for p in c["policies"])


def test_bad_domain_rejected(tmp_path):
    path = tmp_path / "pairs.csv"
    pd.DataFrame([{"policy": "a", "scenario_id": "s", "domain": "lab", "success": 1}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="domain"):
        load_pairs(path)


def test_cli_sweep_compare_regress(tmp_path):
    cfg = tmp_path / "sweep.yaml"
    cfg.write_text("seed: 2\nepisodes: 60\nparams:\n  obstacle: {choice: [false, true]}\n", encoding="utf-8")
    base, cand = tmp_path / "base", tmp_path / "cand"
    assert main(["sweep", "--policy", "pd", "--config", str(cfg), "--out", str(base), "--export-suite"]) == 0
    assert main(["sweep", "--policy", "pd-avoid", "--config", str(cfg), "--out", str(cand)]) == 0
    summary = json.loads((base / "summary.json").read_text(encoding="utf-8"))
    assert summary["episodes"] == 60 and "attribution" in summary
    assert (base / "report.md").read_text(encoding="utf-8").startswith("# Closed-loop evaluation")

    assert main(["compare", str(base), str(cand)]) == 0       # candidate better: no regression
    assert main(["compare", str(cand), str(base)]) == 1       # swapped: regression, non-zero exit
    assert main(["regress", "--policy", "pd-avoid", "--suite", str(base / "suite.json")]) == 0

    # A gate the naive policy cannot meet fails the command.
    assert main(["sweep", "--policy", "pd", "--config", str(cfg), "--out", str(tmp_path / "g"), "--min-success", "0.9"]) == 1
