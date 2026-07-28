# b5a-schlong

![b5a-schlong progress](b5a-schlong.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 35000, avg score 65.6, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b5a-schlong |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.99 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
| initial_epsilon | 0.4 |
| min_epsilon | 0.0 |
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.8 |
| priority_signal | td_loss |
| importance_sampling_beta | disabled |
| initial_populate_steps | 1000 |
| initialize_with_schmid | False |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |

## Evals

36 evals so far. Full series in [`b5a-schlong_evals.json`](b5a-schlong_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.902 | 0 | 0.4 |
| 1000 | 1.4 | 1.4 | 0 | 5/95 | -3.612 | 0 | 0.4 |
| 2000 | 0.0 | 0.7 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| ... | | | | | | | |
| 24000 | 21.7 | 12.62 | 11 | 36/95 | 17.493 | 0 | 0.1 |
| 25000 | 22.4 | 15.8 | 13 | 38/95 | 17.711 | 0 | 0.1 |
| 26000 | 28.2 | 19.72 | 13 | 49/95 | 23.039 | 0 | 0.05 |
| 27000 | 41.7 | 25.34 | 18 | 67/95 | 36.731 | 0 | 0.05 |
| 28000 | 35.3 | 29.86 | 11 | 61/95 | 30.867 | 0 | 0.05 |
| 29000 | 35.5 | 32.62 | 10 | 68/95 | 32.41 | 0 | 0.05 |
| 30000 | 51.2 | 38.38 | 21 | 65/95 | 46.988 | 0 | 0.01 |
| 31000 | 57.5 | 44.24 | 39 | 75/95 | 53.973 | 0 | 0.01 |
| 32000 | 60.7 | 48.04 | 47 | 69/95 | 57.353 | 0 | 0.01 |
| 33000 | 55.8 | 52.14 | 10 | 84/95 | 51.964 | 0 | 0.01 |
| 34000 | 66.8 | 58.4 | 21 | 89/95 | 63.073 | 0 | 0.001 |
| 35000 | 65.6 | 61.28 | 28 | 83/95 | 61.476 | 0 | 0.001 |
