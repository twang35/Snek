# b8d-disc995clip

![b8d-disc995clip progress](b8d-disc995clip.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2099000, avg score 66.1, perfect games 0%.

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

2100 evals so far. Full series in [`b8d-disc995clip_evals.json`](b8d-disc995clip_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.5 | 0.5 | 0 | 2/95 | -3.652 | 0 | 0.4 |
| 2000 | 0.1 | 0.3 | 0 | 1/95 | -4.93 | 0 | 0.4 |
| ... | | | | | | | |
| 2088000 | 81.3 | 76.04 | 50 | 95/95 | 116.99 | 40 | 0.0 |
| 2089000 | 76.8 | 74.3 | 40 | 95/95 | 80.638 | 10 | 0.0 |
| 2090000 | 70.2 | 73.18 | 57 | 95/95 | 74.715 | 10 | 0.0 |
| 2091000 | 75.7 | 73.4 | 52 | 95/95 | 100.969 | 30 | 0.0 |
| 2092000 | 70.3 | 74.86 | 62 | 87/95 | 64.405 | 0 | 0.0 |
| 2093000 | 78.6 | 74.32 | 41 | 95/95 | 103.601 | 30 | 0.0 |
| 2094000 | 71.5 | 73.26 | 25 | 95/95 | 86.422 | 20 | 0.0 |
| 2095000 | 71.9 | 73.6 | 35 | 95/95 | 76.576 | 10 | 0.0 |
| 2096000 | 66.7 | 71.8 | 43 | 95/95 | 71.255 | 10 | 0.0 |
| 2097000 | 90.0 | 75.74 | 70 | 95/95 | 125.855 | 40 | 0.0 |
| 2098000 | 57.2 | 71.46 | 21 | 76/95 | 51.402 | 0 | 0.0 |
| 2099000 | 66.1 | 70.38 | 42 | 86/95 | 59.051 | 0 | 0.0 |
