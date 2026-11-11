# Twin credibility per scenario class and the twin certificate

A single sim-vs-real correlation number hides where a twin can be trusted. A twin can rank checkpoints well on tabletop picks and badly on articulated objects, and one global number averages the two into something that is true of neither. `groot-twin credibility` judges the twin **separately for each scenario class**. It then writes a **certificate** that says, for this twin version and this robot's approved profile, which classes sim results may stand in for robot runs in.

Code: [`credibility.py`](../src/groot_eval/credibility.py). Tests: [`test_credibility.py`](../tests/test_credibility.py).

## Input

A CSV of paired results: the same policies in the twin and on the robot.

| Column | Meaning |
|---|---|
| `policy` | checkpoint or policy name |
| `domain` | `sim` or `real` |
| `scenario_class` | any label: task family, object type, skill (`--class-column` to use another column) |
| `success` | one episode per row, or `successes` and `trials` per aggregate |
| `scenario_id` | optional: pairs a sim episode with the real episode of the same scenario |

`groot-eval pairs` already writes most of these columns. Add the class column.

## Per class

| Metric | Question it answers |
|---|---|
| Pearson r, Spearman | Do sim and real success rates move together? |
| MMRV | How badly does the twin mis-rank two policies, in success-rate units? |
| Bias (sim − real) with an interval | Is the twin optimistic or pessimistic? |
| Pass/fail agreement at `--gate` | Would gating on the twin make the robot's calls? |
| Paired outcome agreement | On the same scenario, do sim and real end the same way? |

Each class gets a verdict:

- **trusted**: every threshold is met and there is enough evidence. Sim may replace robot runs in this class for the decisions the certificate covers.
- **provisional**: thresholds are met but the evidence is thin, for one of these reasons:
  - fewer than 5 policies;
  - fewer than 20 robot episodes for a policy;
  - real success rates span less than 10 points.

  Screen in sim, confirm on the robot.
- **not_trusted**: a threshold failed. Do not use sim results in this class for decisions.

The default thresholds are in `DEFAULT_THRESHOLDS`. Override them with `--threshold max_mmrv=0.05`. They are starting points, not validated limits.

## Commands

```bash
# assess, write credibility.md/json and the certificate; fail CI unless tabletop is trusted
groot-twin credibility results.csv --gate 0.6 \
    --twin arm3-twin --twin-version 1.4.0 \
    --profile registry:profiles:arm-07@approved \
    --require tabletop --out runs/credibility

# before using a sim result as evidence: is this class certified for this robot and twin?
groot-twin cert-check runs/credibility/twin_certificate.json --class tabletop \
    --profile registry:profiles:arm-07@approved --twin-version 1.4.0
```

## What the certificate binds

- **Twin version.** A new twin release needs a new certificate.
- **Robot profile hash.** If the robot is recalibrated or ages into a new profile, the certificate no longer applies (`cert-check` refuses it).
- **Evidence hash.** The certificate names the exact results it was issued on.
- **Expiry.** The default is 90 days (`--valid-days`). Re-run the paired evaluation before it lapses.
- **Decisions covered.** The defaults are checkpoint ranking and regression screening (`--decision`).

## Where it fits

It builds on the robot profiles ([robot-profiles.md](robot-profiles.md)) and the real-to-sim cell checks (`groot-eval real2sim`). Those make the twin match one robot; credibility measures, per class, whether that twin then predicts the robot's results. It refines gate G6 of the transfer verification ([transfer-verification.md](transfer-verification.md)) from one global number to one verdict per class.
