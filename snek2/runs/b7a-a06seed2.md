# b7a-a06seed2

![b7a-a06seed2 progress](b7a-a06seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 715000, avg score 75.2, perfect games 10%.

## Config

| setting | value |
|---|---|
| policy_name | b7a-a06seed2 |
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
| priority_exponent (alpha) | 0.6 |
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

716 evals so far. Full series in [`b7a-a06seed2_evals.json`](b7a-a06seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -5.005 | 0 | 0.4 |
| 2000 | 1.1 | 0.55 | 0 | 4/95 | 0.543 | 0 | 0.4 |
| ... | | | | | | | |
| 704000 | 59.8 | 65.06 | 22 | 76/95 | 54.44 | 0 | 0.0 |
| 705000 | 71.2 | 66.5 | 50 | 92/95 | 65.603 | 0 | 0.0 |
| 706000 | 67.4 | 65.6 | 32 | 95/95 | 72.272 | 10 | 0.0 |
| 707000 | 63.7 | 65.54 | 34 | 95/95 | 68.579 | 10 | 0.0 |
| 708000 | 73.9 | 67.2 | 38 | 95/95 | 78.674 | 10 | 0.0 |
| 709000 | 69.3 | 69.1 | 40 | 92/95 | 63.805 | 0 | 0.0 |
| 710000 | 70.4 | 68.94 | 45 | 95/95 | 75.185 | 10 | 0.0 |
| 711000 | 64.6 | 68.38 | 30 | 88/95 | 59.079 | 0 | 0.0 |
| 712000 | 71.4 | 69.92 | 52 | 88/95 | 65.892 | 0 | 0.0 |
| 713000 | 74.8 | 70.1 | 48 | 93/95 | 69.239 | 0 | 0.0 |
| 714000 | 55.8 | 67.4 | 18 | 90/95 | 50.43 | 0 | 0.0 |
| 715000 | 75.2 | 68.36 | 52 | 95/95 | 80.019 | 10 | 0.0 |
