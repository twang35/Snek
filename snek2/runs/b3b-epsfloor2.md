# b3b-epsfloor2

![b3b-epsfloor2 progress](b3b-epsfloor2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 45000, avg score 45.0, perfect games 0%.

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

46 evals so far. Full series in [`b3b-epsfloor2_evals.json`](b3b-epsfloor2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.902 | 0 | 0.4 |
| 1000 | 1.8 | 1.8 | 0 | 7/95 | -3.234 | 0 | 0.4 |
| 2000 | 0.9 | 1.35 | 0 | 3/95 | -4.116 | 0 | 0.4 |
| ... | | | | | | | |
| 34000 | 70.0 | 60.42 | 37 | 95/95 | 75.651 | 10 | 0.001 |
| 35000 | 70.1 | 61.7 | 44 | 86/95 | 66.251 | 0 | 0.001 |
| 36000 | 51.4 | 60.58 | 19 | 90/95 | 47.303 | 0 | 0.001 |
| 37000 | 55.7 | 62.1 | 38 | 72/95 | 51.627 | 0 | 0.001 |
| 38000 | 50.6 | 59.56 | 3 | 95/95 | 57.331 | 10 | 0.001 |
| 39000 | 49.5 | 55.46 | 18 | 84/95 | 44.548 | 0 | 0.001 |
| 40000 | 53.2 | 52.08 | 11 | 82/95 | 49.169 | 0 | 0.001 |
| 41000 | 50.1 | 51.82 | 23 | 71/95 | 46.453 | 0 | 0.001 |
| 42000 | 57.9 | 52.26 | 7 | 79/95 | 54.066 | 0 | 0.001 |
| 43000 | 51.0 | 52.34 | 17 | 81/95 | 47.675 | 0 | 0.001 |
| 44000 | 47.3 | 51.9 | 5 | 91/95 | 44.106 | 0 | 0.001 |
| 45000 | 45.0 | 50.26 | 7 | 81/95 | 41.852 | 0 | 0.001 |
