# b5c-schlongIS

![b5c-schlongIS progress](b5c-schlongIS.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 37000, avg score 60.7, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b5c-schlongIS |
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
| priority_exponent (alpha) | 0.8 |
| priority_signal | td_loss |
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

38 evals so far. Full series in [`b5c-schlongIS_evals.json`](b5c-schlongIS_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.904 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 1/95 | -5.005 | 0 | 0.4 |
| 2000 | 0.0 | 0.0 | 0 | 0/95 | -5.015 | 0 | 0.4 |
| ... | | | | | | | |
| 26000 | 47.6 | 32.28 | 13 | 64/95 | 42.778 | 0 | 0.01 |
| 27000 | 44.1 | 37.26 | 9 | 70/95 | 39.659 | 0 | 0.01 |
| 28000 | 36.0 | 40.34 | 20 | 58/95 | 31.518 | 0 | 0.01 |
| 29000 | 11.2 | 36.48 | 1 | 44/95 | 9.634 | 0 | 0.01 |
| 30000 | 20.3 | 31.84 | 2 | 56/95 | 18.195 | 0 | 0.01 |
| 31000 | 14.1 | 25.14 | 1 | 42/95 | 12.432 | 0 | 0.01 |
| 32000 | 16.1 | 19.54 | 1 | 74/95 | 14.443 | 0 | 0.01 |
| 33000 | 46.5 | 21.64 | 2 | 73/95 | 42.645 | 0 | 0.01 |
| 34000 | 40.5 | 27.5 | 5 | 77/95 | 37.671 | 0 | 0.01 |
| 35000 | 59.0 | 35.24 | 45 | 70/95 | 54.302 | 0 | 0.01 |
| 36000 | 63.1 | 45.04 | 52 | 71/95 | 58.436 | 0 | 0.01 |
| 37000 | 60.7 | 53.96 | 30 | 80/95 | 56.454 | 0 | 0.01 |
