"""Write a SYNTHETIC sim/real pairs CSV to show the `correlate` command's input and output.

The numbers are invented: five checkpoints, 20 scenarios, sim results that are
optimistic and noisier for some scenarios than others. Replace with real
paired runs (same checkpoint, same scenario, sim and robot) to use it for real.

    python examples/make_example_pairs.py examples/sim_real_pairs_SYNTHETIC.csv
"""

import csv
import sys

import numpy as np

rng = np.random.default_rng(3)
policies = {"ckpt_a": 0.55, "ckpt_b": 0.62, "ckpt_c": 0.70, "ckpt_d": 0.74, "ckpt_e": 0.81}  # "true" real-world skill
scenarios = [f"pick_place_{i:02d}" for i in range(20)]
difficulty = {s: rng.normal(0, 0.15) for s in scenarios}
sim_gap = {s: (0.25 if i in (3, 7, 11) else 0.08) for i, s in enumerate(scenarios)}  # a few badly modelled scenes

out = sys.argv[1] if len(sys.argv) > 1 else "sim_real_pairs_SYNTHETIC.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["policy", "scenario_id", "domain", "success"])
    for policy, skill in policies.items():
        for s in scenarios:
            p_real = float(np.clip(skill - difficulty[s], 0.02, 0.98))
            p_sim = float(np.clip(p_real + sim_gap[s], 0.02, 0.98))
            for _ in range(3):
                w.writerow([policy, s, "sim", int(rng.random() < p_sim)])
            w.writerow([policy, s, "real", int(rng.random() < p_real)])
print(f"wrote {out}")
