# b3c-buf500k

![b3c-buf500k progress](b3c-buf500k.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 26000, avg score 55.8, perfect games 0%.

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

27 evals so far. Full series in [`b3c-buf500k_evals.json`](b3c-buf500k_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.6 | 0.6 | 0 | 3/95 | -4.451 | 0 | 0.4 |
| 2000 | 0.2 | 0.4 | 0 | 1/95 | -4.846 | 0 | 0.4 |
| ... | | | | | | | |
| 15000 | 20.3 | 15.48 | 16 | 25/95 | 15.663 | 0 | 0.1 |
| 16000 | 23.5 | 17.92 | 16 | 30/95 | 19.288 | 0 | 0.1 |
| 17000 | 29.8 | 21.12 | 8 | 43/95 | 24.503 | 0 | 0.05 |
| 18000 | 26.4 | 23.66 | 16 | 39/95 | 22.04 | 0 | 0.05 |
| 19000 | 34.1 | 26.82 | 19 | 47/95 | 30.064 | 0 | 0.05 |
| 20000 | 37.1 | 30.18 | 27 | 50/95 | 32.469 | 0 | 0.05 |
| 21000 | 43.4 | 34.16 | 24 | 54/95 | 39.013 | 0 | 0.05 |
| 22000 | 41.5 | 36.5 | 21 | 58/95 | 36.507 | 0 | 0.05 |
| 23000 | 52.3 | 41.68 | 32 | 74/95 | 47.154 | 0 | 0.01 |
| 24000 | 47.6 | 44.38 | 19 | 65/95 | 43.49 | 0 | 0.01 |
| 25000 | 50.5 | 47.06 | 34 | 68/95 | 46.336 | 0 | 0.01 |
| 26000 | 55.8 | 49.54 | 36 | 75/95 | 50.765 | 0 | 0.01 |
