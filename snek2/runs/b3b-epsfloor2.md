# b3b-epsfloor2

![b3b-epsfloor2 progress](b3b-epsfloor2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 315000, avg score 49.6, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b3b-epsfloor2 |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.99 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
| initial_epsilon | 0.4 |
| min_epsilon | 0.001 |
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

316 evals so far. Full series in [`b3b-epsfloor2_evals.json`](b3b-epsfloor2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.902 | 0 | 0.4 |
| 1000 | 1.8 | 1.8 | 0 | 7/95 | -3.234 | 0 | 0.4 |
| 2000 | 0.9 | 1.35 | 0 | 3/95 | -4.116 | 0 | 0.4 |
| ... | | | | | | | |
| 304000 | 74.1 | 70.7 | 35 | 94/95 | 68.466 | 0 | 0.001 |
| 305000 | 85.8 | 72.28 | 65 | 93/95 | 80.023 | 0 | 0.001 |
| 306000 | 71.6 | 72.8 | 40 | 89/95 | 66.038 | 0 | 0.001 |
| 307000 | 63.4 | 72.72 | 22 | 93/95 | 57.881 | 0 | 0.001 |
| 308000 | 70.2 | 73.02 | 23 | 95/95 | 75.04 | 10 | 0.001 |
| 309000 | 52.9 | 68.78 | 19 | 84/95 | 47.507 | 0 | 0.001 |
| 310000 | 63.3 | 64.28 | 23 | 95/95 | 68.212 | 10 | 0.001 |
| 311000 | 62.0 | 62.36 | 5 | 90/95 | 56.439 | 0 | 0.001 |
| 312000 | 64.8 | 62.64 | 34 | 90/95 | 59.312 | 0 | 0.001 |
| 313000 | 65.8 | 61.76 | 22 | 95/95 | 70.684 | 10 | 0.001 |
| 314000 | 73.0 | 65.78 | 21 | 92/95 | 67.39 | 0 | 0.001 |
| 315000 | 49.6 | 63.04 | 19 | 85/95 | 44.211 | 0 | 0.001 |
