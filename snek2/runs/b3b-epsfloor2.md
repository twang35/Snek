# b3b-epsfloor2

![b3b-epsfloor2 progress](b3b-epsfloor2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 249000, avg score 85.0, perfect games 20%.

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

250 evals so far. Full series in [`b3b-epsfloor2_evals.json`](b3b-epsfloor2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.902 | 0 | 0.4 |
| 1000 | 1.8 | 1.8 | 0 | 7/95 | -3.234 | 0 | 0.4 |
| 2000 | 0.9 | 1.35 | 0 | 3/95 | -4.116 | 0 | 0.4 |
| ... | | | | | | | |
| 238000 | 75.2 | 69.88 | 60 | 95/95 | 80.104 | 10 | 0.001 |
| 239000 | 77.4 | 72.78 | 53 | 93/95 | 71.691 | 0 | 0.001 |
| 240000 | 65.4 | 72.56 | 23 | 89/95 | 59.871 | 0 | 0.001 |
| 241000 | 69.8 | 72.28 | 29 | 95/95 | 74.625 | 10 | 0.001 |
| 242000 | 60.8 | 69.72 | 13 | 95/95 | 65.767 | 10 | 0.001 |
| 243000 | 68.4 | 68.36 | 23 | 92/95 | 63.265 | 0 | 0.001 |
| 244000 | 73.1 | 67.5 | 57 | 95/95 | 77.864 | 10 | 0.001 |
| 245000 | 64.7 | 67.36 | 14 | 95/95 | 69.561 | 10 | 0.001 |
| 246000 | 74.9 | 68.38 | 60 | 95/95 | 79.706 | 10 | 0.001 |
| 247000 | 72.1 | 70.64 | 27 | 95/95 | 76.921 | 10 | 0.001 |
| 248000 | 67.8 | 70.52 | 56 | 83/95 | 62.165 | 0 | 0.001 |
| 249000 | 85.0 | 72.9 | 70 | 95/95 | 100.003 | 20 | 0.001 |
