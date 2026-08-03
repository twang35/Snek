# b10c-disc995seed3

![b10c-disc995seed3 progress](b10c-disc995seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 31000, avg score 30.0, perfect games 0%.

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

32 evals so far. Full series in [`b10c-disc995seed3_evals.json`](b10c-disc995seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 1.9 | 1.9 | 0 | 5/95 | 0.89 | 0 | 0.4 |
| 2000 | 2.7 | 2.3 | 0 | 8/95 | 1.691 | 0 | 0.4 |
| ... | | | | | | | |
| 20000 | 72.7 | 73.58 | 61 | 81/95 | 71.334 | 0 | 0.001 |
| 21000 | 76.0 | 74.34 | 62 | 84/95 | 74.558 | 0 | 0.001 |
| 22000 | 59.1 | 70.74 | 21 | 84/95 | 57.846 | 0 | 0.001 |
| 23000 | 52.7 | 66.4 | 3 | 81/95 | 51.447 | 0 | 0.001 |
| 24000 | 52.4 | 62.58 | 3 | 91/95 | 51.165 | 0 | 0.001 |
| 25000 | 64.9 | 61.02 | 4 | 90/95 | 63.502 | 0 | 0.001 |
| 26000 | 50.7 | 55.96 | 3 | 87/95 | 49.429 | 0 | 0.001 |
| 27000 | 9.6 | 46.06 | 0 | 78/95 | 8.902 | 0 | 0.001 |
| 28000 | 1.5 | 35.82 | 0 | 9/95 | 0.933 | 0 | 0.001 |
| 29000 | 9.7 | 27.28 | 0 | 84/95 | 9.001 | 0 | 0.001 |
| 30000 | 40.1 | 22.32 | 0 | 79/95 | 38.969 | 0 | 0.001 |
| 31000 | 30.0 | 18.18 | 2 | 89/95 | 28.991 | 0 | 0.001 |
