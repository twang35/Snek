# b3c-buf500k

![b3c-buf500k progress](b3c-buf500k.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 232000, avg score 64.1, perfect games 0%.

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

233 evals so far. Full series in [`b3c-buf500k_evals.json`](b3c-buf500k_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.6 | 0.6 | 0 | 3/95 | -4.451 | 0 | 0.4 |
| 2000 | 0.2 | 0.4 | 0 | 1/95 | -4.846 | 0 | 0.4 |
| ... | | | | | | | |
| 221000 | 72.5 | 70.34 | 34 | 95/95 | 77.273 | 10 | 0.001 |
| 222000 | 65.8 | 68.98 | 33 | 91/95 | 60.269 | 0 | 0.001 |
| 223000 | 67.9 | 67.1 | 46 | 83/95 | 62.395 | 0 | 0.001 |
| 224000 | 73.3 | 69.88 | 59 | 86/95 | 67.67 | 0 | 0.001 |
| 225000 | 73.4 | 70.58 | 54 | 88/95 | 67.834 | 0 | 0.001 |
| 226000 | 68.8 | 69.84 | 32 | 86/95 | 63.281 | 0 | 0.001 |
| 227000 | 69.8 | 70.64 | 48 | 93/95 | 64.258 | 0 | 0.001 |
| 228000 | 66.9 | 70.44 | 17 | 89/95 | 61.224 | 0 | 0.001 |
| 229000 | 67.9 | 69.36 | 49 | 90/95 | 62.388 | 0 | 0.001 |
| 230000 | 73.9 | 69.46 | 32 | 95/95 | 78.711 | 10 | 0.001 |
| 231000 | 62.6 | 68.22 | 38 | 81/95 | 57.15 | 0 | 0.001 |
| 232000 | 64.1 | 67.08 | 42 | 84/95 | 58.656 | 0 | 0.001 |
