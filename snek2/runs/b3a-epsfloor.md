# b3a-epsfloor

![b3a-epsfloor progress](b3a-epsfloor.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 44000, avg score 60.2, perfect games 0%.

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

45 evals so far. Full series in [`b3a-epsfloor_evals.json`](b3a-epsfloor_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.2 | 0.2 | 0 | 1/95 | -4.849 | 0 | 0.4 |
| 2000 | 0.3 | 0.25 | 0 | 2/95 | -4.754 | 0 | 0.4 |
| ... | | | | | | | |
| 33000 | 56.1 | 56.34 | 12 | 78/95 | 51.058 | 0 | 0.01 |
| 34000 | 47.0 | 53.92 | 7 | 77/95 | 42.902 | 0 | 0.01 |
| 35000 | 43.8 | 51.26 | 11 | 72/95 | 39.682 | 0 | 0.01 |
| 36000 | 52.0 | 50.46 | 19 | 79/95 | 47.765 | 0 | 0.01 |
| 37000 | 48.2 | 49.42 | 6 | 80/95 | 44.466 | 0 | 0.01 |
| 38000 | 44.7 | 47.14 | 12 | 83/95 | 41.062 | 0 | 0.01 |
| 39000 | 50.6 | 47.86 | 19 | 73/95 | 48.169 | 0 | 0.01 |
| 40000 | 55.6 | 50.22 | 32 | 76/95 | 52.322 | 0 | 0.01 |
| 41000 | 38.5 | 47.52 | 7 | 56/95 | 36.279 | 0 | 0.01 |
| 42000 | 52.1 | 48.3 | 15 | 70/95 | 48.058 | 0 | 0.01 |
| 43000 | 45.7 | 48.5 | 26 | 63/95 | 41.713 | 0 | 0.01 |
| 44000 | 60.2 | 50.42 | 27 | 77/95 | 56.461 | 0 | 0.01 |
