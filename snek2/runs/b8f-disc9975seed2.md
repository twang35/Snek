# b8f-disc9975seed2

![b8f-disc9975seed2 progress](b8f-disc9975seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3029000, avg score 90.7, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | b8f-disc9975seed2 |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.9975 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
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

3030 evals so far. Full series in [`b8f-disc9975seed2_evals.json`](b8f-disc9975seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.3 | 0.3 | 0 | 1/95 | -4.703 | 0 | 0.4 |
| 1000 | 11.3 | 11.3 | 0 | 17/95 | 6.262 | 0 | 0.2 |
| 2000 | 0.1 | 5.7 | 0 | 1/95 | -4.906 | 0 | 0.2 |
| ... | | | | | | | |
| 3018000 | 78.7 | 87.46 | 6 | 95/95 | 124.382 | 50 | 0.0 |
| 3019000 | 81.7 | 84.98 | 4 | 95/95 | 137.584 | 60 | 0.0 |
| 3020000 | 84.6 | 83.56 | 12 | 95/95 | 140.43 | 60 | 0.0 |
| 3021000 | 63.4 | 78.4 | 0 | 95/95 | 88.292 | 30 | 0.0 |
| 3022000 | 89.3 | 79.54 | 83 | 95/95 | 124.241 | 40 | 0.0 |
| 3023000 | 83.5 | 80.5 | 66 | 95/95 | 97.654 | 20 | 0.0 |
| 3024000 | 91.0 | 82.36 | 78 | 95/95 | 157.362 | 70 | 0.0 |
| 3025000 | 86.8 | 82.8 | 60 | 95/95 | 132.192 | 50 | 0.0 |
| 3026000 | 91.2 | 88.36 | 74 | 95/95 | 157.291 | 70 | 0.0 |
| 3027000 | 83.2 | 87.14 | 0 | 95/95 | 159.979 | 80 | 0.0 |
| 3028000 | 70.2 | 84.48 | 0 | 95/95 | 105.36 | 40 | 0.0 |
| 3029000 | 90.7 | 84.42 | 74 | 95/95 | 146.405 | 60 | 0.0 |
