# b6a-alpha04

![b6a-alpha04 progress](b6a-alpha04.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 389000, avg score 80.6, perfect games 10%.

## Config

| setting | value |
|---|---|
| policy_name | b6a-alpha04 |
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
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.4 |
| priority_signal | td_loss |
| importance_sampling_beta | disabled |
| initial_populate_steps | 1000 |
| initialize_with_schmid | False |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |

## Evals

390 evals so far. Full series in [`b6a-alpha04_evals.json`](b6a-alpha04_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -5.053 | 0 | 0.4 |
| 2000 | 0.0 | 0.0 | 0 | 0/95 | -1.001 | 0 | 0.4 |
| ... | | | | | | | |
| 378000 | 61.0 | 69.74 | 15 | 84/95 | 55.502 | 0 | 0.0 |
| 379000 | 83.8 | 71.38 | 74 | 95/95 | 88.366 | 10 | 0.0 |
| 380000 | 70.8 | 72.98 | 28 | 95/95 | 85.954 | 20 | 0.0 |
| 381000 | 72.8 | 73.5 | 13 | 95/95 | 77.552 | 10 | 0.0 |
| 382000 | 81.8 | 74.04 | 27 | 93/95 | 75.879 | 0 | 0.0 |
| 383000 | 73.9 | 76.62 | 38 | 95/95 | 78.639 | 10 | 0.0 |
| 384000 | 83.6 | 76.58 | 60 | 95/95 | 98.643 | 20 | 0.0 |
| 385000 | 71.5 | 76.72 | 15 | 92/95 | 65.747 | 0 | 0.0 |
| 386000 | 77.5 | 77.66 | 34 | 95/95 | 113.421 | 40 | 0.0 |
| 387000 | 85.5 | 78.4 | 60 | 95/95 | 100.452 | 20 | 0.0 |
| 388000 | 83.5 | 80.32 | 34 | 95/95 | 119.439 | 40 | 0.0 |
| 389000 | 80.6 | 79.72 | 48 | 95/95 | 85.321 | 10 | 0.0 |
