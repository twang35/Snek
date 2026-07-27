# b3b-epsfloor2

![b3b-epsfloor2 progress](b3b-epsfloor2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 36000, avg score 51.4, perfect games 0%.

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

37 evals so far. Full series in [`b3b-epsfloor2_evals.json`](b3b-epsfloor2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.902 | 0 | 0.4 |
| 1000 | 1.8 | 1.8 | 0 | 7/95 | -3.234 | 0 | 0.4 |
| 2000 | 0.9 | 1.35 | 0 | 3/95 | -4.116 | 0 | 0.4 |
| ... | | | | | | | |
| 25000 | 38.9 | 22.84 | 22 | 52/95 | 33.754 | 0 | 0.05 |
| 26000 | 57.6 | 31.48 | 30 | 79/95 | 52.25 | 0 | 0.01 |
| 27000 | 60.7 | 39.96 | 30 | 86/95 | 55.278 | 0 | 0.01 |
| 28000 | 55.2 | 46.92 | 13 | 74/95 | 50.226 | 0 | 0.01 |
| 29000 | 57.0 | 53.88 | 30 | 82/95 | 52.372 | 0 | 0.01 |
| 30000 | 63.7 | 58.84 | 1 | 85/95 | 59.425 | 0 | 0.01 |
| 31000 | 57.0 | 58.72 | 12 | 75/95 | 52.849 | 0 | 0.01 |
| 32000 | 48.1 | 56.2 | 2 | 79/95 | 45.385 | 0 | 0.01 |
| 33000 | 63.3 | 57.82 | 49 | 69/95 | 58.529 | 0 | 0.01 |
| 34000 | 70.0 | 60.42 | 37 | 95/95 | 75.651 | 10 | 0.001 |
| 35000 | 70.1 | 61.7 | 44 | 86/95 | 66.251 | 0 | 0.001 |
| 36000 | 51.4 | 60.58 | 19 | 90/95 | 47.303 | 0 | 0.001 |
