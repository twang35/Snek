# b3a-epsfloor

![b3a-epsfloor progress](b3a-epsfloor.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 319000, avg score 66.2, perfect games 0%.

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

320 evals so far. Full series in [`b3a-epsfloor_evals.json`](b3a-epsfloor_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.2 | 0.2 | 0 | 1/95 | -4.849 | 0 | 0.4 |
| 2000 | 0.3 | 0.25 | 0 | 2/95 | -4.754 | 0 | 0.4 |
| ... | | | | | | | |
| 308000 | 73.7 | 71.42 | 37 | 95/95 | 88.808 | 20 | 0.001 |
| 309000 | 78.2 | 72.58 | 54 | 95/95 | 82.93 | 10 | 0.001 |
| 310000 | 78.3 | 73.42 | 47 | 95/95 | 83.109 | 10 | 0.001 |
| 311000 | 72.1 | 74.28 | 41 | 94/95 | 66.334 | 0 | 0.001 |
| 312000 | 69.0 | 74.26 | 44 | 95/95 | 73.949 | 10 | 0.001 |
| 313000 | 68.7 | 73.26 | 32 | 95/95 | 73.487 | 10 | 0.001 |
| 314000 | 71.9 | 72.0 | 55 | 82/95 | 66.219 | 0 | 0.001 |
| 315000 | 76.7 | 71.68 | 56 | 93/95 | 71.064 | 0 | 0.001 |
| 316000 | 69.5 | 71.16 | 33 | 95/95 | 84.79 | 20 | 0.001 |
| 317000 | 67.0 | 70.76 | 31 | 91/95 | 61.367 | 0 | 0.001 |
| 318000 | 79.8 | 72.98 | 57 | 95/95 | 84.525 | 10 | 0.001 |
| 319000 | 66.2 | 71.84 | 31 | 91/95 | 60.71 | 0 | 0.001 |
