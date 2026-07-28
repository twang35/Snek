# b5d-schlongTDE

![b5d-schlongTDE progress](b5d-schlongTDE.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 535000, avg score 68.4, perfect games 10%.

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

536 evals so far. Full series in [`b5d-schlongTDE_evals.json`](b5d-schlongTDE_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.004 | 0 | 0.4 |
| 1000 | 0.5 | 0.5 | 0 | 1/95 | -0.054 | 0 | 0.4 |
| 2000 | 0.3 | 0.4 | 0 | 1/95 | -1.143 | 0 | 0.4 |
| ... | | | | | | | |
| 524000 | 70.2 | 69.98 | 46 | 88/95 | 64.659 | 0 | 0.0 |
| 525000 | 67.3 | 71.54 | 10 | 95/95 | 82.551 | 20 | 0.0 |
| 526000 | 79.9 | 72.44 | 57 | 93/95 | 74.182 | 0 | 0.0 |
| 527000 | 63.1 | 70.24 | 27 | 87/95 | 57.515 | 0 | 0.0 |
| 528000 | 76.2 | 71.34 | 65 | 94/95 | 70.581 | 0 | 0.0 |
| 529000 | 67.8 | 70.86 | 36 | 95/95 | 72.694 | 10 | 0.0 |
| 530000 | 72.9 | 71.98 | 34 | 95/95 | 77.65 | 10 | 0.0 |
| 531000 | 67.8 | 69.56 | 46 | 93/95 | 62.305 | 0 | 0.0 |
| 532000 | 70.3 | 71.0 | 60 | 95/95 | 75.16 | 10 | 0.0 |
| 533000 | 62.7 | 68.3 | 20 | 79/95 | 57.226 | 0 | 0.0 |
| 534000 | 58.3 | 66.4 | 1 | 94/95 | 52.717 | 0 | 0.0 |
| 535000 | 68.4 | 65.5 | 45 | 95/95 | 73.331 | 10 | 0.0 |
