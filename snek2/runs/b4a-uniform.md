# b4a-uniform

![b4a-uniform progress](b4a-uniform.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1294000, avg score 55.2, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b4a-uniform |
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
| priority_exponent (alpha) | 0.0 |
| priority_signal | td_error |
| importance_sampling_beta | 0.4 -> 1.0 over 1000000 steps |
| initial_populate_steps | 1000 |
| initialize_with_schmid | False |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |

## Evals

1295 evals so far. Full series in [`b4a-uniform_evals.json`](b4a-uniform_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.904 | 0 | 0.4 |
| 1000 | 0.6 | 0.6 | 0 | 3/95 | -0.867 | 0 | 0.4 |
| 2000 | 3.0 | 1.8 | 0 | 6/95 | -1.635 | 0 | 0.4 |
| ... | | | | | | | |
| 1283000 | 57.9 | 67.98 | 26 | 83/95 | 52.529 | 0 | 0.0 |
| 1284000 | 75.6 | 68.66 | 43 | 95/95 | 80.45 | 10 | 0.0 |
| 1285000 | 64.7 | 66.48 | 37 | 86/95 | 59.255 | 0 | 0.0 |
| 1286000 | 70.1 | 68.76 | 44 | 85/95 | 64.684 | 0 | 0.0 |
| 1287000 | 69.3 | 67.52 | 28 | 89/95 | 63.803 | 0 | 0.0 |
| 1288000 | 59.9 | 67.92 | 28 | 95/95 | 75.328 | 20 | 0.0 |
| 1289000 | 60.3 | 64.86 | 38 | 78/95 | 54.921 | 0 | 0.0 |
| 1290000 | 55.0 | 62.92 | 21 | 95/95 | 60.081 | 10 | 0.0 |
| 1291000 | 70.1 | 62.92 | 33 | 88/95 | 64.577 | 0 | 0.0 |
| 1292000 | 67.0 | 62.46 | 25 | 87/95 | 61.518 | 0 | 0.0 |
| 1293000 | 69.0 | 64.28 | 35 | 91/95 | 63.595 | 0 | 0.0 |
| 1294000 | 55.2 | 63.26 | 24 | 94/95 | 49.868 | 0 | 0.0 |
