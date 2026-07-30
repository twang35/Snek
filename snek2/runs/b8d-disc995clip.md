# b8d-disc995clip

![b8d-disc995clip progress](b8d-disc995clip.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 938000, avg score 73.4, perfect games 10%.

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

939 evals so far. Full series in [`b8d-disc995clip_evals.json`](b8d-disc995clip_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.5 | 0.5 | 0 | 2/95 | -3.652 | 0 | 0.4 |
| 2000 | 0.1 | 0.3 | 0 | 1/95 | -4.93 | 0 | 0.4 |
| ... | | | | | | | |
| 927000 | 56.6 | 73.0 | 11 | 95/95 | 61.346 | 10 | 0.0 |
| 928000 | 59.7 | 69.98 | 23 | 87/95 | 53.878 | 0 | 0.0 |
| 929000 | 85.8 | 71.82 | 57 | 95/95 | 111.134 | 30 | 0.0 |
| 930000 | 78.0 | 73.88 | 43 | 95/95 | 92.94 | 20 | 0.0 |
| 931000 | 76.5 | 71.32 | 55 | 95/95 | 80.992 | 10 | 0.0 |
| 932000 | 69.9 | 73.98 | 40 | 88/95 | 64.225 | 0 | 0.0 |
| 933000 | 77.7 | 77.58 | 60 | 93/95 | 71.736 | 0 | 0.0 |
| 934000 | 57.4 | 71.9 | 11 | 82/95 | 51.79 | 0 | 0.0 |
| 935000 | 75.9 | 71.48 | 35 | 95/95 | 111.597 | 40 | 0.0 |
| 936000 | 72.8 | 70.74 | 47 | 95/95 | 77.358 | 10 | 0.0 |
| 937000 | 66.7 | 70.1 | 17 | 95/95 | 81.733 | 20 | 0.0 |
| 938000 | 73.4 | 69.24 | 44 | 95/95 | 78.002 | 10 | 0.0 |
