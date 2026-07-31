# b8d-disc995clip

![b8d-disc995clip progress](b8d-disc995clip.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3484000, avg score 68.3, perfect games 70%.

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

3485 evals so far. Full series in [`b8d-disc995clip_evals.json`](b8d-disc995clip_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.5 | 0.5 | 0 | 2/95 | -3.652 | 0 | 0.4 |
| 2000 | 0.1 | 0.3 | 0 | 1/95 | -4.93 | 0 | 0.4 |
| ... | | | | | | | |
| 3473000 | 85.8 | 80.78 | 72 | 95/95 | 100.539 | 20 | 0.0 |
| 3474000 | 86.9 | 82.8 | 64 | 95/95 | 122.535 | 40 | 0.0 |
| 3475000 | 85.1 | 82.76 | 74 | 95/95 | 120.69 | 40 | 0.0 |
| 3476000 | 72.8 | 80.86 | 3 | 95/95 | 77.32 | 10 | 0.0 |
| 3477000 | 79.2 | 81.96 | 21 | 95/95 | 104.287 | 30 | 0.0 |
| 3478000 | 66.0 | 78.0 | 3 | 95/95 | 91.224 | 30 | 0.0 |
| 3479000 | 78.7 | 76.36 | 10 | 95/95 | 93.368 | 20 | 0.0 |
| 3480000 | 80.3 | 75.4 | 5 | 95/95 | 105.374 | 30 | 0.0 |
| 3481000 | 78.8 | 76.6 | 4 | 95/95 | 93.584 | 20 | 0.0 |
| 3482000 | 61.8 | 73.12 | 1 | 95/95 | 87.202 | 30 | 0.0 |
| 3483000 | 48.1 | 69.54 | 2 | 95/95 | 63.153 | 20 | 0.0 |
| 3484000 | 68.3 | 67.46 | 4 | 95/95 | 135.241 | 70 | 0.0 |
