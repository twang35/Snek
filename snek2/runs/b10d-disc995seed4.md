# b10d-disc995seed4

![b10d-disc995seed4 progress](b10d-disc995seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1225000, avg score 88.0, perfect games 80%.

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

1226 evals so far. Full series in [`b10d-disc995seed4_evals.json`](b10d-disc995seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.004 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 2/95 | -0.152 | 0 | 0.4 |
| 2000 | 0.0 | 0.2 | 0 | 0/95 | -0.548 | 0 | 0.4 |
| ... | | | | | | | |
| 1214000 | 92.0 | 90.2 | 77 | 95/95 | 139.105 | 50 | 0.0 |
| 1215000 | 88.4 | 89.48 | 51 | 95/95 | 145.986 | 60 | 0.0 |
| 1216000 | 82.4 | 88.98 | 45 | 95/95 | 109.977 | 30 | 0.0 |
| 1217000 | 90.0 | 89.0 | 80 | 95/95 | 126.298 | 40 | 0.0 |
| 1218000 | 92.6 | 89.08 | 84 | 95/95 | 160.723 | 70 | 0.0 |
| 1219000 | 86.1 | 87.9 | 53 | 95/95 | 143.723 | 60 | 0.0 |
| 1220000 | 80.7 | 86.36 | 39 | 95/95 | 128.075 | 50 | 0.0 |
| 1221000 | 93.4 | 88.56 | 84 | 95/95 | 170.808 | 80 | 0.0 |
| 1222000 | 91.5 | 88.86 | 82 | 95/95 | 109.188 | 20 | 0.0 |
| 1223000 | 93.8 | 89.1 | 88 | 95/95 | 161.19 | 70 | 0.0 |
| 1224000 | 93.6 | 90.6 | 81 | 95/95 | 182.053 | 90 | 0.0 |
| 1225000 | 88.0 | 92.06 | 49 | 95/95 | 166.436 | 80 | 0.0 |
