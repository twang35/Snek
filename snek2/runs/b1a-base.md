# b1a-base

![b1a-base progress](b1a-base.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 86000, avg score 75.2, perfect games 10%.

## Config

| setting | value |
|---|---|
| policy_name | b1a-base |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.99 |
| target_update_period | 8 |
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

87 evals so far. Full series in [`b1a-base_evals.json`](b1a-base_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.005 | 0 | 0.4 |
| 1000 | 2.9 | 2.9 | 0 | 8/95 | -2.155 | 0 | 0.4 |
| 2000 | 5.1 | 4.0 | 0 | 12/95 | 0.08 | 0 | 0.4 |
| ... | | | | | | | |
| 75000 | 74.5 | 68.04 | 53 | 93/95 | 70.394 | 0 | 0.001 |
| 76000 | 68.1 | 68.82 | 7 | 91/95 | 64.166 | 0 | 0.001 |
| 77000 | 65.6 | 67.98 | 36 | 95/95 | 71.279 | 10 | 0.001 |
| 78000 | 72.8 | 69.12 | 28 | 87/95 | 69.666 | 0 | 0.001 |
| 79000 | 64.3 | 69.06 | 30 | 95/95 | 69.564 | 10 | 0.001 |
| 80000 | 70.3 | 68.22 | 48 | 86/95 | 66.771 | 0 | 0.001 |
| 81000 | 59.4 | 66.48 | 35 | 89/95 | 56.791 | 0 | 0.001 |
| 82000 | 67.5 | 66.86 | 38 | 85/95 | 63.081 | 0 | 0.001 |
| 83000 | 68.7 | 66.04 | 42 | 84/95 | 65.183 | 0 | 0.001 |
| 84000 | 71.3 | 67.44 | 48 | 93/95 | 66.482 | 0 | 0.001 |
| 85000 | 68.8 | 67.14 | 40 | 91/95 | 64.756 | 0 | 0.001 |
| 86000 | 75.2 | 70.3 | 53 | 95/95 | 81.556 | 10 | 0.001 |
