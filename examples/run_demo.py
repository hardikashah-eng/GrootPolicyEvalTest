"""Run the whole workflow on the built-in simulator and write every report.

    uv run python examples/run_demo.py [output_dir]

Takes well under a minute on a laptop CPU.
"""

import shutil
import sys
from pathlib import Path

from groot_eval.cli import main

ROOT = Path(__file__).resolve().parent.parent
out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "demo-output")
shutil.rmtree(out, ignore_errors=True)
cfg = str(ROOT / "configs" / "reach2d_sweep.yaml")


def step(title, argv, expect=0):
    print(f"\n=== {title}\n$ groot-eval {' '.join(argv)}")
    code = main(argv)
    if code != expect:
        raise SystemExit(f"step failed: exit {code}, expected {expect}")


step("Baseline checkpoint: naive controller", ["sweep", "--policy", "pd", "--config", cfg, "--out", str(out / "baseline"), "--export-suite"])
step("Candidate checkpoint: obstacle-aware controller", ["sweep", "--policy", "pd-avoid", "--config", cfg, "--out", str(out / "candidate")])
step("Is the candidate significantly better?", ["compare", str(out / "baseline"), str(out / "candidate"), "--out", str(out / "compare.md")])
step("Replay the baseline's failures against the candidate", ["regress", "--policy", "pd-avoid", "--suite", str(out / "baseline" / "suite.json"), "--out", str(out / "regress.md")])
step("Action chunking: execute 16 steps open-loop vs re-plan every 2",
     ["sweep", "--policy", "pd-chunk", "--policy-arg", "execute_steps=16", "--config", cfg, "--out", str(out / "chunk16")])
step("", ["sweep", "--policy", "pd-chunk", "--policy-arg", "execute_steps=2", "--config", cfg, "--out", str(out / "chunk2")])
step("Record demonstrations (LeRobot v2 format)", ["make-demo-data", "--out", str(out / "dataset"), "--episodes", "30"])
step("Open-loop: predicted chunks vs recorded actions", ["open-loop", "--dataset", str(out / "dataset"), "--policy", "pd-chunk", "--horizon", "16", "--out", str(out / "open_loop")])
pairs = out / "sim_real_pairs_SYNTHETIC.csv"
import runpy  # noqa: E402

sys.argv = ["make_example_pairs.py", str(pairs)]
runpy.run_path(str(ROOT / "examples" / "make_example_pairs.py"), run_name="__main__")
step("Sim-to-real correlation (synthetic data)", ["correlate", str(pairs), "--out", str(out / "correlation.md")])
step("How many episodes do we need?", ["plan", "--margin", "0.05", "--expected", "0.8"])
print(f"\nAll reports are in {out}")
