# b3a-epsfloor

![b3a-epsfloor progress](b3a-epsfloor.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 33000, avg score 56.1, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b3a-epsfloor |
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

34 evals so far. Full series in [`b3a-epsfloor_evals.json`](b3a-epsfloor_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.2 | 0.2 | 0 | 1/95 | -4.849 | 0 | 0.4 |
| 2000 | 0.3 | 0.25 | 0 | 2/95 | -4.754 | 0 | 0.4 |
| ... | | | | | | | |
| 22000 | 51.5 | 36.7 | 25 | 63/95 | 48.297 | 0 | 0.01 |
| 23000 | 58.6 | 44.74 | 10 | 79/95 | 56.511 | 0 | 0.01 |
| 24000 | 51.0 | 50.28 | 20 | 76/95 | 46.465 | 0 | 0.01 |
| 25000 | 57.3 | 54.08 | 19 | 90/95 | 52.737 | 0 | 0.01 |
| 26000 | 56.2 | 54.92 | 14 | 72/95 | 51.248 | 0 | 0.01 |
| 27000 | 55.4 | 55.7 | 11 | 74/95 | 50.854 | 0 | 0.01 |
| 28000 | 47.8 | 53.54 | 7 | 81/95 | 44.136 | 0 | 0.01 |
| 29000 | 59.1 | 55.16 | 44 | 83/95 | 54.003 | 0 | 0.01 |
| 30000 | 57.1 | 55.12 | 20 | 78/95 | 51.996 | 0 | 0.01 |
| 31000 | 56.0 | 55.08 | 34 | 75/95 | 51.459 | 0 | 0.01 |
| 32000 | 53.4 | 54.68 | 28 | 71/95 | 49.242 | 0 | 0.01 |
| 33000 | 56.1 | 56.34 | 12 | 78/95 | 51.058 | 0 | 0.01 |
