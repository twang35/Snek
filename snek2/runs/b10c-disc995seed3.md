# b10c-disc995seed3

![b10c-disc995seed3 progress](b10c-disc995seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1060000, avg score 91.5, perfect games 50%.

## Config

| setting | value |
|---|---|
| policy_name | b10c-disc995seed3 |
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

1061 evals so far. Full series in [`b10c-disc995seed3_evals.json`](b10c-disc995seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 1.9 | 1.9 | 0 | 5/95 | 0.89 | 0 | 0.4 |
| 2000 | 2.7 | 2.3 | 0 | 8/95 | 1.691 | 0 | 0.4 |
| ... | | | | | | | |
| 1049000 | 88.7 | 86.5 | 73 | 95/95 | 127.224 | 40 | 0.0 |
| 1050000 | 87.9 | 86.64 | 55 | 95/95 | 116.098 | 30 | 0.0 |
| 1051000 | 89.6 | 88.94 | 69 | 95/95 | 117.684 | 30 | 0.0 |
| 1052000 | 81.3 | 87.32 | 51 | 95/95 | 119.501 | 40 | 0.0 |
| 1053000 | 87.3 | 86.96 | 77 | 95/95 | 105.812 | 20 | 0.0 |
| 1054000 | 85.3 | 86.28 | 67 | 95/95 | 103.33 | 20 | 0.0 |
| 1055000 | 78.8 | 84.46 | 57 | 93/95 | 77.059 | 0 | 0.0 |
| 1056000 | 87.5 | 84.04 | 73 | 95/95 | 115.616 | 30 | 0.0 |
| 1057000 | 79.6 | 83.7 | 31 | 95/95 | 107.697 | 30 | 0.0 |
| 1058000 | 88.6 | 83.96 | 75 | 95/95 | 96.213 | 10 | 0.0 |
| 1059000 | 80.0 | 82.9 | 25 | 95/95 | 108.072 | 30 | 0.0 |
| 1060000 | 91.5 | 85.44 | 83 | 95/95 | 139.992 | 50 | 0.0 |
