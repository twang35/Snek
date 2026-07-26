# b1a-base

![b1a-base progress](b1a-base.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 68000, avg score 70.3, perfect games 0%.

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

69 evals so far. Full series in [`b1a-base_evals.json`](b1a-base_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.005 | 0 | 0.4 |
| 1000 | 2.9 | 2.9 | 0 | 8/95 | -2.155 | 0 | 0.4 |
| 2000 | 5.1 | 4.0 | 0 | 12/95 | 0.08 | 0 | 0.4 |
| ... | | | | | | | |
| 57000 | 66.2 | 67.5 | 31 | 93/95 | 63.498 | 0 | 0.001 |
| 58000 | 65.4 | 66.8 | 42 | 90/95 | 60.638 | 0 | 0.001 |
| 59000 | 70.3 | 69.1 | 38 | 90/95 | 65.849 | 0 | 0.001 |
| 60000 | 75.5 | 68.48 | 60 | 84/95 | 70.065 | 0 | 0.001 |
| 61000 | 68.5 | 69.18 | 48 | 84/95 | 64.305 | 0 | 0.001 |
| 62000 | 67.0 | 69.34 | 35 | 89/95 | 63.805 | 0 | 0.001 |
| 63000 | 67.3 | 69.72 | 45 | 84/95 | 62.979 | 0 | 0.001 |
| 64000 | 73.5 | 70.36 | 49 | 91/95 | 69.934 | 0 | 0.001 |
| 65000 | 70.6 | 69.38 | 17 | 91/95 | 67.821 | 0 | 0.001 |
| 66000 | 67.6 | 69.2 | 27 | 95/95 | 74.081 | 10 | 0.001 |
| 67000 | 70.3 | 69.86 | 38 | 86/95 | 66.804 | 0 | 0.001 |
| 68000 | 70.3 | 70.46 | 50 | 88/95 | 65.468 | 0 | 0.001 |
