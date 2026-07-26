# b1b-tgt200

![b1b-tgt200 progress](b1b-tgt200.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 106000, avg score 68.7, perfect games 0%.

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

107 evals so far. Full series in [`b1b-tgt200_evals.json`](b1b-tgt200_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 1.1 | 1.1 | 0 | 3/95 | -3.915 | 0 | 0.4 |
| 2000 | 1.8 | 1.45 | 0 | 4/95 | -1.004 | 0 | 0.4 |
| ... | | | | | | | |
| 95000 | 65.7 | 66.62 | 26 | 95/95 | 71.099 | 10 | 0.001 |
| 96000 | 61.5 | 65.56 | 39 | 81/95 | 57.301 | 0 | 0.001 |
| 97000 | 62.1 | 66.0 | 30 | 89/95 | 57.533 | 0 | 0.001 |
| 98000 | 50.4 | 61.78 | 23 | 86/95 | 45.076 | 0 | 0.001 |
| 99000 | 57.2 | 59.38 | 26 | 76/95 | 52.193 | 0 | 0.001 |
| 100000 | 64.9 | 59.22 | 48 | 95/95 | 69.832 | 10 | 0.001 |
| 101000 | 64.0 | 59.72 | 48 | 89/95 | 58.991 | 0 | 0.001 |
| 102000 | 54.8 | 58.26 | 26 | 74/95 | 50.348 | 0 | 0.001 |
| 103000 | 68.1 | 61.8 | 28 | 89/95 | 64.738 | 0 | 0.001 |
| 104000 | 60.1 | 62.38 | 22 | 89/95 | 55.512 | 0 | 0.001 |
| 105000 | 60.8 | 61.56 | 14 | 91/95 | 55.828 | 0 | 0.001 |
| 106000 | 68.7 | 62.5 | 48 | 92/95 | 64.507 | 0 | 0.001 |
