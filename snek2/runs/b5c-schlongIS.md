# b5c-schlongIS

![b5c-schlongIS progress](b5c-schlongIS.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 571000, avg score 76.3, perfect games 10%.

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

572 evals so far. Full series in [`b5c-schlongIS_evals.json`](b5c-schlongIS_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.904 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 1/95 | -5.005 | 0 | 0.4 |
| 2000 | 0.0 | 0.0 | 0 | 0/95 | -5.015 | 0 | 0.4 |
| ... | | | | | | | |
| 560000 | 67.1 | 73.58 | 28 | 87/95 | 61.601 | 0 | 0.0 |
| 561000 | 58.1 | 69.88 | 28 | 81/95 | 52.745 | 0 | 0.0 |
| 562000 | 77.2 | 69.04 | 48 | 89/95 | 71.593 | 0 | 0.0 |
| 563000 | 72.8 | 68.64 | 8 | 95/95 | 77.555 | 10 | 0.0 |
| 564000 | 59.0 | 66.84 | 26 | 90/95 | 53.536 | 0 | 0.0 |
| 565000 | 70.8 | 67.58 | 42 | 86/95 | 65.291 | 0 | 0.0 |
| 566000 | 70.2 | 70.0 | 48 | 85/95 | 64.753 | 0 | 0.0 |
| 567000 | 82.5 | 71.06 | 65 | 95/95 | 87.114 | 10 | 0.0 |
| 568000 | 70.2 | 70.54 | 42 | 88/95 | 64.645 | 0 | 0.0 |
| 569000 | 75.8 | 73.9 | 49 | 95/95 | 80.668 | 10 | 0.0 |
| 570000 | 68.6 | 73.46 | 50 | 87/95 | 62.984 | 0 | 0.0 |
| 571000 | 76.3 | 74.68 | 52 | 95/95 | 81.094 | 10 | 0.0 |
