# b5c-schlongIS

![b5c-schlongIS progress](b5c-schlongIS.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 998000, avg score 62.8, perfect games 0%.

Training was resumed at step 40000 (the dashed lines on the graph).

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

999 evals so far. Full series in [`b5c-schlongIS_evals.json`](b5c-schlongIS_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.904 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 1/95 | -5.005 | 0 | 0.4 |
| 2000 | 0.0 | 0.0 | 0 | 0/95 | -5.015 | 0 | 0.4 |
| ... | | | | | | | |
| 987000 | 59.8 | 64.32 | 14 | 95/95 | 64.779 | 10 | 0.0 |
| 988000 | 61.5 | 64.92 | 35 | 88/95 | 56.034 | 0 | 0.0 |
| 989000 | 59.9 | 63.72 | 30 | 87/95 | 54.509 | 0 | 0.0 |
| 990000 | 76.5 | 64.14 | 36 | 95/95 | 81.318 | 10 | 0.0 |
| 991000 | 57.1 | 62.96 | 29 | 84/95 | 51.678 | 0 | 0.0 |
| 992000 | 61.4 | 63.28 | 23 | 95/95 | 66.387 | 10 | 0.0 |
| 993000 | 65.2 | 64.02 | 45 | 88/95 | 59.748 | 0 | 0.0 |
| 994000 | 75.8 | 67.2 | 32 | 95/95 | 80.662 | 10 | 0.0 |
| 995000 | 66.0 | 65.1 | 28 | 95/95 | 70.953 | 10 | 0.0 |
| 996000 | 72.9 | 68.26 | 29 | 95/95 | 88.15 | 20 | 0.0 |
| 997000 | 69.4 | 69.86 | 15 | 92/95 | 63.865 | 0 | 0.0 |
| 998000 | 62.8 | 69.38 | 31 | 90/95 | 57.41 | 0 | 0.0 |
