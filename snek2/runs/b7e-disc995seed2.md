# b7e-disc995seed2

![b7e-disc995seed2 progress](b7e-disc995seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 60000, avg score 64.8, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b7e-disc995seed2 |
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
| initialize_with_schmid | False |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |

## Evals

61 evals so far. Full series in [`b7e-disc995seed2_evals.json`](b7e-disc995seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.004 | 0 | 0.4 |
| 1000 | 1.2 | 1.2 | 0 | 4/95 | -3.812 | 0 | 0.4 |
| 2000 | 1.3 | 1.25 | 0 | 7/95 | -3.711 | 0 | 0.4 |
| ... | | | | | | | |
| 49000 | 34.5 | 25.14 | 11 | 58/95 | 29.827 | 0 | 0.05 |
| 50000 | 30.7 | 27.56 | 5 | 42/95 | 26.05 | 0 | 0.05 |
| 51000 | 25.6 | 28.08 | 16 | 40/95 | 20.523 | 0 | 0.05 |
| 52000 | 34.1 | 30.7 | 9 | 46/95 | 28.975 | 0 | 0.05 |
| 53000 | 36.8 | 32.34 | 14 | 61/95 | 31.637 | 0 | 0.05 |
| 54000 | 38.5 | 33.14 | 2 | 68/95 | 33.306 | 0 | 0.05 |
| 55000 | 42.1 | 35.42 | 13 | 74/95 | 36.802 | 0 | 0.05 |
| 56000 | 47.1 | 39.72 | 20 | 79/95 | 41.806 | 0 | 0.01 |
| 57000 | 55.9 | 44.08 | 24 | 90/95 | 50.534 | 0 | 0.01 |
| 58000 | 42.2 | 45.16 | 15 | 71/95 | 36.982 | 0 | 0.01 |
| 59000 | 48.6 | 47.18 | 22 | 74/95 | 43.308 | 0 | 0.01 |
| 60000 | 64.8 | 51.72 | 12 | 84/95 | 59.809 | 0 | 0.01 |
