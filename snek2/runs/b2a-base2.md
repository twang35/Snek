# b2a-base2

![b2a-base2 progress](b2a-base2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 699000, avg score 61.6, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b2a-base2 |
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

700 evals so far. Full series in [`b2a-base2_evals.json`](b2a-base2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 1/95 | -0.153 | 0 | 0.4 |
| 2000 | 1.1 | 0.75 | 0 | 4/95 | 0.543 | 0 | 0.4 |
| ... | | | | | | | |
| 688000 | 65.5 | 67.22 | 15 | 84/95 | 59.988 | 0 | 0.001 |
| 689000 | 62.6 | 64.74 | 17 | 92/95 | 57.157 | 0 | 0.001 |
| 690000 | 79.2 | 66.7 | 57 | 90/95 | 73.58 | 0 | 0.001 |
| 691000 | 64.8 | 67.0 | 32 | 85/95 | 59.35 | 0 | 0.001 |
| 692000 | 74.1 | 69.24 | 55 | 95/95 | 78.942 | 10 | 0.001 |
| 693000 | 56.6 | 67.46 | 13 | 95/95 | 61.616 | 10 | 0.001 |
| 694000 | 70.1 | 68.96 | 35 | 95/95 | 74.919 | 10 | 0.001 |
| 695000 | 71.6 | 67.44 | 51 | 90/95 | 66.02 | 0 | 0.001 |
| 696000 | 66.1 | 67.7 | 34 | 94/95 | 60.641 | 0 | 0.001 |
| 697000 | 61.9 | 65.26 | 29 | 82/95 | 56.474 | 0 | 0.001 |
| 698000 | 63.8 | 66.7 | 34 | 88/95 | 58.363 | 0 | 0.001 |
| 699000 | 61.6 | 65.0 | 15 | 89/95 | 56.086 | 0 | 0.001 |
