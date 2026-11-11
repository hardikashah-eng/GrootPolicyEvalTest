"""Fleet sweep on a batched toy plant: how does tracking degrade with age, unit by unit?

Each joint is a pendulum-like link (inertia, gravity load) driven by the
:class:`WearAwareActuatorCore`. All ``ages x units`` environments step together
in one array, which is the same batched path Isaac Lab uses, so this doubles as
a check of a wear profile before a GPU run: if a profile makes a 3-year-old
unit unable to hold its own arm, it shows here in seconds.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from groot_eval.actuators.core import ActuatorParams, FaultTable, WearAwareActuatorCore, WearTable


def run_sweep(wear: WearTable, params: ActuatorParams, faults: FaultTable | None = None, steps: int = 2000,
              inertia: float = 0.5, gravity_torque: float = 4.0, amplitude: float = 0.6,
              period_s: float = 2.0, joint_limit: float = 2.5) -> dict[str, Any]:
    e = wear.num_envs
    j = next(iter(wear.values.values())).shape[1]
    act = WearAwareActuatorCore(params, wear, faults)
    dt = params.dt
    q = np.zeros((e, j))
    qd = np.zeros((e, j))
    err2 = np.zeros((e, j))
    seen = np.zeros((e, j))
    sat = np.zeros((e, j))
    limit = np.broadcast_to(np.asarray(params.effort_limit, dtype=float), (j,))
    for k in range(steps):
        t = k * dt
        target = np.full((e, j), amplitude * np.sin(2 * np.pi * t / period_s))
        target_v = np.full((e, j), amplitude * 2 * np.pi / period_s * np.cos(2 * np.pi * t / period_s))
        tau = act.compute(target, target_v, gravity_torque * np.sin(target), q, qd)
        q_meas = act.observe(q)
        ok = np.isfinite(q_meas)  # dropout faults report NaN; score only real readings
        err2 += np.where(ok, (np.where(ok, q_meas, 0.0) - target) ** 2, 0.0)
        seen += ok
        sat += np.abs(tau) >= 0.999 * limit * wear.values["torque_limit"]
        qdd = (tau - gravity_torque * np.sin(q)) / inertia
        qd = qd + dt * qdd
        q = q + dt * qd
        hit = np.abs(q) > joint_limit  # hard stop: a runaway motor pins the joint instead of spinning forever
        q = np.clip(q, -joint_limit, joint_limit)
        qd = np.where(hit, 0.0, qd)
    rmse = np.sqrt(err2 / np.maximum(seen, 1)).mean(axis=1)
    rows = []
    base = None
    for age in np.unique(wear.ages_hours):
        m = wear.ages_hours == age
        r = rmse[m]
        row = {"age_hours": float(age), "units": int(m.sum()), "rmse_mean": float(r.mean()),
               "rmse_p90": float(np.percentile(r, 90)), "rmse_worst": float(r.max()),
               "saturated_frac": float(sat[m].mean() / steps), "heat_mean": float(act.heat[m].mean())}
        if base is None:
            base = (wear.unit_seeds[m], r)
        else:
            common = {s: v for s, v in zip(*base)}
            diffs = [v - common[s] for s, v in zip(wear.unit_seeds[m], r) if s in common]
            row["paired_increase_mean"] = float(np.mean(diffs)) if diffs else float("nan")
            row["units_worse"] = int(sum(d > 0 for d in diffs))
        rows.append(row)
    return {"rows": rows, "per_env_rmse": rmse.tolist(), "ages": wear.ages_hours.tolist(),
            "seeds": wear.unit_seeds.tolist(), "steps": steps, "dt": dt}


def sweep_markdown(result: dict[str, Any], title: str) -> str:
    lines = [f"# Actuator fleet sweep: {title}", "",
             "Tracking a sine on a gravity-loaded link, every unit at every age in one batch. The same units appear at "
             "every age, so increases are paired.", "",
             "| Age (h) | Units | RMSE mean (rad) | p90 | Worst | Paired increase vs new | Units worse | Saturated | Motor heat |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in result["rows"]:
        inc = r.get("paired_increase_mean")
        lines.append(f"| {r['age_hours']:g} | {r['units']} | {r['rmse_mean']:.4f} | {r['rmse_p90']:.4f} | "
                     f"{r['rmse_worst']:.4f} | {'–' if inc is None else f'{inc:+.4f}'} | "
                     f"{r.get('units_worse', '–')} | {100 * r['saturated_frac']:.1f}% | {r['heat_mean']:.2f} |")
    return "\n".join(lines) + "\n"
