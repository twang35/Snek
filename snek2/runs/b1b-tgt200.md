# b1b-tgt200

![b1b-tgt200 progress](b1b-tgt200.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 83000, avg score 64.4, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b1b-tgt200 |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.99 |
| target_update_period | 200 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
| initial_epsilon | 0.4 |
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

84 evals so far. Full series in [`b1b-tgt200_evals.json`](b1b-tgt200_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 1.1 | 1.1 | 0 | 3/95 | -3.915 | 0 | 0.4 |
| 2000 | 1.8 | 1.45 | 0 | 4/95 | -1.004 | 0 | 0.4 |
| ... | | | | | | | |
| 72000 | 63.6 | 62.24 | 25 | 85/95 | 60.265 | 0 | 0.001 |
| 73000 | 59.7 | 62.5 | 20 | 83/95 | 54.744 | 0 | 0.001 |
| 74000 | 64.2 | 61.64 | 22 | 87/95 | 59.95 | 0 | 0.001 |
| 75000 | 59.3 | 62.42 | 22 | 85/95 | 54.703 | 0 | 0.001 |
| 76000 | 69.7 | 63.3 | 54 | 82/95 | 64.999 | 0 | 0.001 |
| 77000 | 72.8 | 65.14 | 59 | 78/95 | 67.637 | 0 | 0.001 |
| 78000 | 54.6 | 64.12 | 44 | 86/95 | 50.091 | 0 | 0.001 |
| 79000 | 51.9 | 61.66 | 28 | 68/95 | 46.565 | 0 | 0.001 |
| 80000 | 58.4 | 61.48 | 23 | 81/95 | 53.392 | 0 | 0.001 |
| 81000 | 58.1 | 59.16 | 34 | 86/95 | 53.068 | 0 | 0.001 |
| 82000 | 63.2 | 57.24 | 51 | 71/95 | 58.54 | 0 | 0.001 |
| 83000 | 64.4 | 59.2 | 27 | 89/95 | 59.794 | 0 | 0.001 |
