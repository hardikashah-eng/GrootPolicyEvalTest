"""Degraded robots: wear and tear by age (:mod:`.wear`) and mid-episode faults (:mod:`.faults`).

Both are env wrappers driven by scenario parameters (``wear.*`` and
``faults``), so sweeps, attribution, regression suites and traces work on them
unchanged. Stack them as ``FaultEnv(WearEnv(env))``.
"""

from groot_eval.degradation.faults import FaultConfig, FaultEnv, faulted_scenarios, parse_schedule
from groot_eval.degradation.wear import WearConfig, WearEnv, aged_scenarios

__all__ = ["FaultConfig", "FaultEnv", "WearConfig", "WearEnv", "aged_scenarios", "faulted_scenarios", "parse_schedule"]
