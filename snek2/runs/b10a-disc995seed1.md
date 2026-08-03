# b10a-disc995seed1

![b10a-disc995seed1 progress](b10a-disc995seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1173000, avg score 86.0, perfect games 40%.

## Config

| setting | value |
|---|---|
| policy_name | b10a-disc995seed1 |
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

1174 evals so far. Full series in [`b10a-disc995seed1_evals.json`](b10a-disc995seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.901 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | -2.531 | 0 | 0.4 |
| 2000 | 0.7 | 0.7 | 0 | 2/95 | 0.142 | 0 | 0.4 |
| ... | | | | | | | |
| 1162000 | 93.6 | 87.54 | 87 | 95/95 | 172.009 | 80 | 0.0 |
| 1163000 | 91.2 | 89.48 | 73 | 95/95 | 139.704 | 50 | 0.0 |
| 1164000 | 88.9 | 90.54 | 70 | 95/95 | 126.873 | 40 | 0.0 |
| 1165000 | 89.2 | 90.02 | 75 | 95/95 | 126.711 | 40 | 0.0 |
| 1166000 | 80.4 | 88.66 | 55 | 95/95 | 98.534 | 20 | 0.0 |
| 1167000 | 87.4 | 87.42 | 71 | 95/95 | 115.418 | 30 | 0.0 |
| 1168000 | 90.8 | 87.34 | 77 | 95/95 | 138.735 | 50 | 0.0 |
| 1169000 | 83.9 | 86.34 | 28 | 95/95 | 122.405 | 40 | 0.0 |
| 1170000 | 87.8 | 86.06 | 60 | 95/95 | 145.384 | 60 | 0.0 |
| 1171000 | 90.9 | 88.16 | 80 | 95/95 | 138.873 | 50 | 0.0 |
| 1172000 | 93.8 | 89.44 | 86 | 95/95 | 171.974 | 80 | 0.0 |
| 1173000 | 86.0 | 88.48 | 68 | 95/95 | 123.506 | 40 | 0.0 |
