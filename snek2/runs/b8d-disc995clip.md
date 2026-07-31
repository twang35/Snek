# b8d-disc995clip

![b8d-disc995clip progress](b8d-disc995clip.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3502000, avg score 76.7, perfect games 10%.

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

3503 evals so far. Full series in [`b8d-disc995clip_evals.json`](b8d-disc995clip_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.5 | 0.5 | 0 | 2/95 | -3.652 | 0 | 0.4 |
| 2000 | 0.1 | 0.3 | 0 | 1/95 | -4.93 | 0 | 0.4 |
| ... | | | | | | | |
| 3491000 | 60.2 | 58.64 | 3 | 93/95 | 53.992 | 0 | 0.0 |
| 3492000 | 22.1 | 50.06 | 3 | 95/95 | 26.912 | 10 | 0.0 |
| 3493000 | 52.4 | 51.64 | 4 | 95/95 | 56.99 | 10 | 0.0 |
| 3494000 | 52.4 | 45.54 | 4 | 95/95 | 57.124 | 10 | 0.0 |
| 3495000 | 53.2 | 48.06 | 6 | 91/95 | 47.523 | 0 | 0.0 |
| 3496000 | 69.9 | 50.0 | 39 | 95/95 | 84.9 | 20 | 0.0 |
| 3497000 | 70.5 | 59.68 | 5 | 95/95 | 95.973 | 30 | 0.0 |
| 3498000 | 67.4 | 62.68 | 7 | 86/95 | 61.518 | 0 | 0.0 |
| 3499000 | 70.6 | 66.32 | 50 | 94/95 | 64.559 | 0 | 0.0 |
| 3500000 | 74.8 | 70.64 | 47 | 95/95 | 89.749 | 20 | 0.0 |
| 3501000 | 70.8 | 70.82 | 50 | 95/95 | 74.942 | 10 | 0.0 |
| 3502000 | 76.7 | 72.06 | 52 | 95/95 | 80.581 | 10 | 0.0 |
