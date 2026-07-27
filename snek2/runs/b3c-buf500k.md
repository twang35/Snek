# b3c-buf500k

![b3c-buf500k progress](b3c-buf500k.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 299000, avg score 70.2, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b3c-buf500k |
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
| replay_buffer | cpprb prioritized, capacity 500000 |
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

300 evals so far. Full series in [`b3c-buf500k_evals.json`](b3c-buf500k_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.6 | 0.6 | 0 | 3/95 | -4.451 | 0 | 0.4 |
| 2000 | 0.2 | 0.4 | 0 | 1/95 | -4.846 | 0 | 0.4 |
| ... | | | | | | | |
| 288000 | 60.8 | 71.62 | 32 | 90/95 | 55.352 | 0 | 0.0 |
| 289000 | 73.4 | 70.38 | 47 | 93/95 | 68.222 | 0 | 0.0 |
| 290000 | 64.0 | 70.1 | 16 | 91/95 | 58.471 | 0 | 0.0 |
| 291000 | 66.8 | 68.66 | 36 | 89/95 | 61.139 | 0 | 0.0 |
| 292000 | 68.9 | 66.78 | 51 | 94/95 | 63.44 | 0 | 0.0 |
| 293000 | 64.3 | 67.48 | 16 | 93/95 | 58.79 | 0 | 0.0 |
| 294000 | 72.7 | 67.34 | 35 | 95/95 | 77.531 | 10 | 0.0 |
| 295000 | 72.9 | 69.12 | 61 | 94/95 | 67.31 | 0 | 0.0 |
| 296000 | 73.9 | 70.54 | 52 | 89/95 | 68.28 | 0 | 0.0 |
| 297000 | 66.4 | 70.04 | 30 | 87/95 | 61.001 | 0 | 0.0 |
| 298000 | 70.0 | 71.18 | 27 | 92/95 | 64.353 | 0 | 0.0 |
| 299000 | 70.2 | 70.68 | 41 | 91/95 | 64.669 | 0 | 0.0 |
