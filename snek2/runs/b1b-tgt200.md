# b1b-tgt200

![b1b-tgt200 progress](b1b-tgt200.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 69000, avg score 68.5, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b1b-tgt200 |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.99 |
| target_update_period | 200 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
| initial_epsilon | 0.4 |
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
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

70 evals so far. Full series in [`b1b-tgt200_evals.json`](b1b-tgt200_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 1.1 | 1.1 | 0 | 3/95 | -3.915 | 0 | 0.4 |
| 2000 | 1.8 | 1.45 | 0 | 4/95 | -1.004 | 0 | 0.4 |
| ... | | | | | | | |
| 58000 | 49.5 | 61.56 | 19 | 76/95 | 45.414 | 0 | 0.001 |
| 59000 | 71.7 | 62.78 | 41 | 87/95 | 66.791 | 0 | 0.001 |
| 60000 | 68.0 | 64.62 | 48 | 83/95 | 63.238 | 0 | 0.001 |
| 61000 | 60.5 | 63.86 | 40 | 89/95 | 55.834 | 0 | 0.001 |
| 62000 | 62.6 | 62.46 | 43 | 79/95 | 57.841 | 0 | 0.001 |
| 63000 | 58.1 | 64.18 | 17 | 78/95 | 53.482 | 0 | 0.001 |
| 64000 | 62.0 | 62.24 | 41 | 78/95 | 56.919 | 0 | 0.001 |
| 65000 | 59.6 | 60.56 | 22 | 84/95 | 54.919 | 0 | 0.001 |
| 66000 | 61.8 | 60.82 | 46 | 92/95 | 56.356 | 0 | 0.001 |
| 67000 | 54.5 | 59.2 | 28 | 76/95 | 51.29 | 0 | 0.001 |
| 68000 | 58.4 | 59.26 | 28 | 77/95 | 53.03 | 0 | 0.001 |
| 69000 | 68.5 | 60.56 | 47 | 88/95 | 64.214 | 0 | 0.001 |
