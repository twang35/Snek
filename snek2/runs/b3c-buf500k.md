# b3c-buf500k

![b3c-buf500k progress](b3c-buf500k.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 525000, avg score 72.2, perfect games 10%.

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

526 evals so far. Full series in [`b3c-buf500k_evals.json`](b3c-buf500k_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.6 | 0.6 | 0 | 3/95 | -4.451 | 0 | 0.4 |
| 2000 | 0.2 | 0.4 | 0 | 1/95 | -4.846 | 0 | 0.4 |
| ... | | | | | | | |
| 514000 | 72.3 | 69.74 | 56 | 91/95 | 66.725 | 0 | 0.0 |
| 515000 | 71.8 | 69.74 | 48 | 91/95 | 66.194 | 0 | 0.0 |
| 516000 | 69.5 | 69.6 | 36 | 89/95 | 63.999 | 0 | 0.0 |
| 517000 | 67.0 | 69.78 | 52 | 84/95 | 61.506 | 0 | 0.0 |
| 518000 | 72.4 | 70.6 | 55 | 87/95 | 66.767 | 0 | 0.0 |
| 519000 | 73.5 | 70.84 | 25 | 94/95 | 68.297 | 0 | 0.0 |
| 520000 | 70.0 | 70.48 | 46 | 87/95 | 64.526 | 0 | 0.0 |
| 521000 | 65.2 | 69.62 | 50 | 95/95 | 70.201 | 10 | 0.0 |
| 522000 | 70.5 | 70.32 | 48 | 92/95 | 65.036 | 0 | 0.0 |
| 523000 | 68.2 | 69.48 | 40 | 86/95 | 62.678 | 0 | 0.0 |
| 524000 | 69.7 | 68.72 | 38 | 88/95 | 64.184 | 0 | 0.0 |
| 525000 | 72.2 | 69.16 | 41 | 95/95 | 77.149 | 10 | 0.0 |
