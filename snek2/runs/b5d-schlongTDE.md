# b5d-schlongTDE

![b5d-schlongTDE progress](b5d-schlongTDE.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2075000, avg score 73.7, perfect games 0%.

Training was resumed at step 35000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | b5d-schlongTDE |
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
| priority_signal | td_error |
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

2076 evals so far. Full series in [`b5d-schlongTDE_evals.json`](b5d-schlongTDE_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.004 | 0 | 0.4 |
| 1000 | 0.5 | 0.5 | 0 | 1/95 | -0.054 | 0 | 0.4 |
| 2000 | 0.3 | 0.4 | 0 | 1/95 | -1.143 | 0 | 0.4 |
| ... | | | | | | | |
| 2064000 | 70.9 | 74.62 | 19 | 89/95 | 65.38 | 0 | 0.0 |
| 2065000 | 56.7 | 69.24 | 10 | 95/95 | 61.678 | 10 | 0.0 |
| 2066000 | 66.7 | 68.24 | 11 | 86/95 | 61.211 | 0 | 0.0 |
| 2067000 | 84.9 | 69.66 | 41 | 95/95 | 100.006 | 20 | 0.0 |
| 2068000 | 68.6 | 69.56 | 32 | 95/95 | 73.526 | 10 | 0.0 |
| 2069000 | 82.8 | 71.94 | 43 | 95/95 | 97.962 | 20 | 0.0 |
| 2070000 | 67.0 | 74.0 | 29 | 90/95 | 61.542 | 0 | 0.0 |
| 2071000 | 69.1 | 74.48 | 20 | 93/95 | 63.57 | 0 | 0.0 |
| 2072000 | 79.7 | 73.44 | 60 | 93/95 | 74.084 | 0 | 0.0 |
| 2073000 | 78.1 | 75.34 | 15 | 95/95 | 82.821 | 10 | 0.0 |
| 2074000 | 67.9 | 72.36 | 33 | 91/95 | 62.343 | 0 | 0.0 |
| 2075000 | 73.7 | 73.7 | 19 | 93/95 | 68.08 | 0 | 0.0 |
