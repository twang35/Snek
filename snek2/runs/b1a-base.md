# b1a-base

![b1a-base progress](b1a-base.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 503000, avg score 67.1, perfect games 0%.

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

504 evals so far. Full series in [`b1a-base_evals.json`](b1a-base_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.005 | 0 | 0.4 |
| 1000 | 2.9 | 2.9 | 0 | 8/95 | -2.155 | 0 | 0.4 |
| 2000 | 5.1 | 4.0 | 0 | 12/95 | 0.08 | 0 | 0.4 |
| ... | | | | | | | |
| 492000 | 61.4 | 60.48 | 11 | 95/95 | 66.297 | 10 | 0.0 |
| 493000 | 75.3 | 63.08 | 29 | 95/95 | 80.033 | 10 | 0.0 |
| 494000 | 61.4 | 65.26 | 31 | 92/95 | 55.887 | 0 | 0.0 |
| 495000 | 64.3 | 65.02 | 23 | 83/95 | 58.834 | 0 | 0.0 |
| 496000 | 62.6 | 65.0 | 32 | 91/95 | 56.998 | 0 | 0.0 |
| 497000 | 69.1 | 66.54 | 56 | 85/95 | 63.544 | 0 | 0.0 |
| 498000 | 54.0 | 62.28 | 4 | 77/95 | 48.542 | 0 | 0.0 |
| 499000 | 64.2 | 62.84 | 41 | 83/95 | 58.664 | 0 | 0.0 |
| 500000 | 67.4 | 63.46 | 43 | 82/95 | 61.84 | 0 | 0.0 |
| 501000 | 61.2 | 63.18 | 17 | 85/95 | 55.65 | 0 | 0.0 |
| 502000 | 68.5 | 63.06 | 44 | 82/95 | 63.009 | 0 | 0.0 |
| 503000 | 67.1 | 65.68 | 19 | 88/95 | 61.502 | 0 | 0.0 |
