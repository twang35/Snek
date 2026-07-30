# b8f-disc9975seed2

![b8f-disc9975seed2 progress](b8f-disc9975seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 486000, avg score 86.5, perfect games 40%.

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

487 evals so far. Full series in [`b8f-disc9975seed2_evals.json`](b8f-disc9975seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.3 | 0.3 | 0 | 1/95 | -4.703 | 0 | 0.4 |
| 1000 | 11.3 | 11.3 | 0 | 17/95 | 6.262 | 0 | 0.2 |
| 2000 | 0.1 | 5.7 | 0 | 1/95 | -4.906 | 0 | 0.2 |
| ... | | | | | | | |
| 475000 | 63.5 | 74.64 | 1 | 95/95 | 68.359 | 10 | 0.0 |
| 476000 | 71.2 | 74.58 | 3 | 95/95 | 75.945 | 10 | 0.0 |
| 477000 | 82.3 | 74.1 | 31 | 95/95 | 128.613 | 50 | 0.0 |
| 478000 | 73.1 | 73.14 | 3 | 95/95 | 119.518 | 50 | 0.0 |
| 479000 | 67.2 | 71.46 | 21 | 95/95 | 71.947 | 10 | 0.0 |
| 480000 | 92.4 | 77.24 | 76 | 95/95 | 169.849 | 80 | 0.0 |
| 481000 | 67.1 | 76.42 | 2 | 95/95 | 71.913 | 10 | 0.0 |
| 482000 | 78.6 | 75.68 | 2 | 95/95 | 114.48 | 40 | 0.0 |
| 483000 | 80.9 | 77.24 | 19 | 95/95 | 116.771 | 40 | 0.0 |
| 484000 | 40.7 | 71.94 | 2 | 95/95 | 45.659 | 10 | 0.0 |
| 485000 | 86.6 | 70.78 | 72 | 95/95 | 111.888 | 30 | 0.0 |
| 486000 | 86.5 | 74.66 | 58 | 95/95 | 122.251 | 40 | 0.0 |
