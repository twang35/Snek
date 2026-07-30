# b8e-clipseed2

![b8e-clipseed2 progress](b8e-clipseed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1165000, avg score 48.3, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b8e-clipseed2 |
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

1166 evals so far. Full series in [`b8e-clipseed2_evals.json`](b8e-clipseed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.2 | 0.2 | 0 | 1/95 | -4.839 | 0 | 0.4 |
| 2000 | 2.4 | 1.3 | 0 | 8/95 | -2.67 | 0 | 0.4 |
| ... | | | | | | | |
| 1154000 | 50.5 | 50.7 | 10 | 95/95 | 65.581 | 20 | 0.0 |
| 1155000 | 59.3 | 52.0 | 21 | 95/95 | 63.481 | 10 | 0.0 |
| 1156000 | 46.8 | 50.38 | 11 | 92/95 | 40.749 | 0 | 0.0 |
| 1157000 | 47.6 | 49.6 | 14 | 71/95 | 41.805 | 0 | 0.0 |
| 1158000 | 61.7 | 53.18 | 44 | 91/95 | 55.069 | 0 | 0.0 |
| 1159000 | 52.3 | 53.54 | 24 | 84/95 | 46.178 | 0 | 0.0 |
| 1160000 | 60.2 | 53.72 | 25 | 88/95 | 53.981 | 0 | 0.0 |
| 1161000 | 49.9 | 54.34 | 14 | 80/95 | 43.708 | 0 | 0.0 |
| 1162000 | 55.7 | 55.96 | 39 | 74/95 | 49.554 | 0 | 0.0 |
| 1163000 | 58.6 | 55.34 | 17 | 95/95 | 62.269 | 10 | 0.0 |
| 1164000 | 64.6 | 57.8 | 47 | 90/95 | 57.985 | 0 | 0.0 |
| 1165000 | 48.3 | 55.42 | 14 | 78/95 | 42.324 | 0 | 0.0 |
