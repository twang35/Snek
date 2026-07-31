# b8f-disc9975seed2

![b8f-disc9975seed2 progress](b8f-disc9975seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3555000, avg score 21.2, perfect games 0%.

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

3556 evals so far. Full series in [`b8f-disc9975seed2_evals.json`](b8f-disc9975seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.3 | 0.3 | 0 | 1/95 | -4.703 | 0 | 0.4 |
| 1000 | 11.3 | 11.3 | 0 | 17/95 | 6.262 | 0 | 0.2 |
| 2000 | 0.1 | 5.7 | 0 | 1/95 | -4.906 | 0 | 0.2 |
| ... | | | | | | | |
| 3544000 | 33.3 | 38.78 | 4 | 82/95 | 26.694 | 0 | 0.0 |
| 3545000 | 46.0 | 39.58 | 8 | 84/95 | 38.587 | 0 | 0.0 |
| 3546000 | 32.8 | 37.34 | 2 | 90/95 | 25.983 | 0 | 0.0 |
| 3547000 | 41.9 | 36.1 | 3 | 95/95 | 44.805 | 10 | 0.0 |
| 3548000 | 33.0 | 37.4 | 3 | 91/95 | 26.19 | 0 | 0.0 |
| 3549000 | 22.1 | 35.16 | 5 | 60/95 | 16.021 | 0 | 0.0 |
| 3550000 | 33.7 | 32.7 | 1 | 84/95 | 26.67 | 0 | 0.0 |
| 3551000 | 28.1 | 31.76 | 1 | 82/95 | 21.435 | 0 | 0.0 |
| 3552000 | 24.1 | 28.2 | 1 | 62/95 | 17.848 | 0 | 0.0 |
| 3553000 | 26.6 | 26.92 | 0 | 95/95 | 30.778 | 10 | 0.0 |
| 3554000 | 30.8 | 28.66 | 3 | 79/95 | 24.169 | 0 | 0.0 |
| 3555000 | 21.2 | 26.16 | 0 | 65/95 | 14.979 | 0 | 0.0 |
