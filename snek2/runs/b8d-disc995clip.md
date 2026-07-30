# b8d-disc995clip

![b8d-disc995clip progress](b8d-disc995clip.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2407000, avg score 66.8, perfect games 10%.

## Config

| setting | value |
|---|---|
| policy_name | b8d-disc995clip |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.995 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | 10.0 |
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
| perfect_game_wait_ms | 500 |

## Evals

2408 evals so far. Full series in [`b8d-disc995clip_evals.json`](b8d-disc995clip_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.5 | 0.5 | 0 | 2/95 | -3.652 | 0 | 0.4 |
| 2000 | 0.1 | 0.3 | 0 | 1/95 | -4.93 | 0 | 0.4 |
| ... | | | | | | | |
| 2396000 | 83.9 | 74.96 | 72 | 95/95 | 98.794 | 20 | 0.0 |
| 2397000 | 67.8 | 76.08 | 50 | 89/95 | 61.931 | 0 | 0.0 |
| 2398000 | 64.5 | 72.22 | 29 | 95/95 | 68.505 | 10 | 0.0 |
| 2399000 | 80.4 | 72.8 | 13 | 95/95 | 115.844 | 40 | 0.0 |
| 2400000 | 71.4 | 73.6 | 48 | 95/95 | 75.85 | 10 | 0.0 |
| 2401000 | 70.9 | 71.0 | 30 | 94/95 | 64.896 | 0 | 0.0 |
| 2402000 | 63.1 | 70.06 | 8 | 95/95 | 88.668 | 30 | 0.0 |
| 2403000 | 74.7 | 72.1 | 9 | 95/95 | 120.772 | 50 | 0.0 |
| 2404000 | 74.2 | 70.86 | 10 | 95/95 | 88.92 | 20 | 0.0 |
| 2405000 | 75.6 | 71.7 | 39 | 95/95 | 80.169 | 10 | 0.0 |
| 2406000 | 84.5 | 74.42 | 66 | 95/95 | 120.082 | 40 | 0.0 |
| 2407000 | 66.8 | 75.16 | 30 | 95/95 | 70.952 | 10 | 0.0 |
