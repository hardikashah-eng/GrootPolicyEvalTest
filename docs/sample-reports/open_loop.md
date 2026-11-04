# Open-loop evaluation: `pd-chunk(h=16)`

Action MSE **0.0140** (95% bootstrap CI 0.0105 – 0.0183) over 30 episodes, horizon 16. MAE 0.0643.

## Error by action group

| Group | MSE |
|---|---|
| end_effector | 0.0147 |

## Error by step within the chunk

Error that grows along the chunk means the policy's internal model drifts from the demonstrator's; it predicts how much executing long chunks open-loop will hurt.

| Step | MSE |
|---|---|
| 0 | 0.0000 |
| 1 | 0.0003 |
| 2 | 0.0009 |
| 3 | 0.0016 |
| 4 | 0.0054 |
| 5 | 0.0096 |
| 6 | 0.0203 |
| 7 | 0.0289 |
| 8 | 0.0307 |
| 9 | 0.0290 |
| 10 | 0.0261 |
| 11 | 0.0243 |
| 12 | 0.0210 |
| 13 | 0.0189 |
| 14 | 0.0160 |
| 15 | 0.0134 |

## Worst episodes

| Episode | Task | MSE | Max abs error |
|---|---|---|---|
| 10 | move the end effector to the target | 0.0608 | 0.895 |
| 14 | move the end effector to the target | 0.0275 | 0.665 |
| 3 | move the end effector to the target | 0.0265 | 0.572 |
| 0 | move the end effector to the target | 0.0218 | 0.635 |
| 23 | move the end effector to the target | 0.0202 | 0.552 |
