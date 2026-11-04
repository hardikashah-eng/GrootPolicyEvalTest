# Checkpoint comparison

**Improvement**: candidate is significantly better.

| Run | Policy | Success | CI | n | Latency p95 (ms) |
|---|---|---|---|---|---|
| baseline | `pd(kp=2.0,kd=1.2)` | 62.3% | 56.7% – 67.6% | 187/300 | 0.01 |
| candidate | `pd(kp=2.0,kd=1.2,avoid=2.0)` | 73.3% | 68.1% – 78.0% | 220/300 | 0.02 |

Two-proportion z = 2.88 (one-sided 95% threshold: ±1.645).

## Failure reasons

| Reason | Baseline | Candidate |
|---|---|---|
| collision | 110 | 77 |
| timeout | 3 | 3 |
