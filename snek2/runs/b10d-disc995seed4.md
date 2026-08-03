# b10d-disc995seed4

![b10d-disc995seed4 progress](b10d-disc995seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1151000, avg score 89.0, perfect games 80%.

## Config

| setting | value |
|---|---|
| policy_name | b10d-disc995seed4 |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.995 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
| initial_epsilon | 0.4 |
| min_epsilon | 0.0 |
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | td_loss |
| importance_sampling_beta | disabled |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

1152 evals so far. Full series in [`b10d-disc995seed4_evals.json`](b10d-disc995seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.004 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 2/95 | -0.152 | 0 | 0.4 |
| 2000 | 0.0 | 0.2 | 0 | 0/95 | -0.548 | 0 | 0.4 |
| ... | | | | | | | |
| 1140000 | 90.3 | 89.84 | 76 | 95/95 | 138.055 | 50 | 0.0 |
| 1141000 | 92.4 | 89.82 | 82 | 95/95 | 159.971 | 70 | 0.0 |
| 1142000 | 83.5 | 89.7 | 31 | 95/95 | 132.018 | 50 | 0.0 |
| 1143000 | 92.7 | 89.6 | 86 | 95/95 | 150.216 | 60 | 0.0 |
| 1144000 | 92.1 | 90.2 | 78 | 95/95 | 170.172 | 80 | 0.0 |
| 1145000 | 88.5 | 89.84 | 75 | 95/95 | 127.084 | 40 | 0.0 |
| 1146000 | 88.6 | 89.08 | 69 | 95/95 | 136.176 | 50 | 0.0 |
| 1147000 | 93.7 | 91.12 | 88 | 95/95 | 161.701 | 70 | 0.0 |
| 1148000 | 91.8 | 90.94 | 84 | 95/95 | 149.87 | 60 | 0.0 |
| 1149000 | 90.7 | 90.66 | 79 | 95/95 | 148.294 | 60 | 0.0 |
| 1150000 | 88.1 | 90.58 | 41 | 95/95 | 156.604 | 70 | 0.0 |
| 1151000 | 89.0 | 90.66 | 43 | 95/95 | 167.491 | 80 | 0.0 |
