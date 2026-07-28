# b6a-alpha04

![b6a-alpha04 progress](b6a-alpha04.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1414000, avg score 74.6, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b6a-alpha04 |
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
| priority_exponent (alpha) | 0.4 |
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

1415 evals so far. Full series in [`b6a-alpha04_evals.json`](b6a-alpha04_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -5.053 | 0 | 0.4 |
| 2000 | 0.0 | 0.0 | 0 | 0/95 | -1.001 | 0 | 0.4 |
| ... | | | | | | | |
| 1403000 | 69.6 | 71.02 | 45 | 82/95 | 63.825 | 0 | 0.0 |
| 1404000 | 75.6 | 71.64 | 62 | 85/95 | 69.555 | 0 | 0.0 |
| 1405000 | 84.3 | 73.34 | 64 | 95/95 | 88.523 | 10 | 0.0 |
| 1406000 | 65.7 | 71.68 | 44 | 80/95 | 60.128 | 0 | 0.0 |
| 1407000 | 76.0 | 74.24 | 0 | 95/95 | 90.848 | 20 | 0.0 |
| 1408000 | 77.0 | 75.72 | 62 | 95/95 | 81.273 | 10 | 0.0 |
| 1409000 | 73.9 | 75.38 | 64 | 86/95 | 68.006 | 0 | 0.0 |
| 1410000 | 77.0 | 73.92 | 44 | 95/95 | 81.476 | 10 | 0.0 |
| 1411000 | 78.8 | 76.54 | 64 | 92/95 | 72.793 | 0 | 0.0 |
| 1412000 | 66.1 | 74.56 | 42 | 77/95 | 60.295 | 0 | 0.0 |
| 1413000 | 70.8 | 73.32 | 9 | 82/95 | 65.064 | 0 | 0.0 |
| 1414000 | 74.6 | 73.46 | 61 | 87/95 | 68.805 | 0 | 0.0 |
