# b8f-disc9975seed2

![b8f-disc9975seed2 progress](b8f-disc9975seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2202000, avg score 84.9, perfect games 40%.

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

2203 evals so far. Full series in [`b8f-disc9975seed2_evals.json`](b8f-disc9975seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.3 | 0.3 | 0 | 1/95 | -4.703 | 0 | 0.4 |
| 1000 | 11.3 | 11.3 | 0 | 17/95 | 6.262 | 0 | 0.2 |
| 2000 | 0.1 | 5.7 | 0 | 1/95 | -4.906 | 0 | 0.2 |
| ... | | | | | | | |
| 2191000 | 84.1 | 85.6 | 76 | 95/95 | 99.098 | 20 | 0.0 |
| 2192000 | 67.6 | 82.2 | 2 | 95/95 | 93.225 | 30 | 0.0 |
| 2193000 | 80.9 | 81.44 | 62 | 95/95 | 95.975 | 20 | 0.0 |
| 2194000 | 80.4 | 79.94 | 45 | 95/95 | 105.909 | 30 | 0.0 |
| 2195000 | 67.8 | 76.16 | 1 | 95/95 | 103.859 | 40 | 0.0 |
| 2196000 | 72.4 | 73.82 | 0 | 95/95 | 87.112 | 20 | 0.0 |
| 2197000 | 63.3 | 72.96 | 2 | 95/95 | 99.414 | 40 | 0.0 |
| 2198000 | 82.3 | 73.24 | 64 | 93/95 | 76.549 | 0 | 0.0 |
| 2199000 | 78.5 | 72.86 | 56 | 95/95 | 93.626 | 20 | 0.0 |
| 2200000 | 78.6 | 75.02 | 50 | 95/95 | 93.768 | 20 | 0.0 |
| 2201000 | 87.6 | 78.06 | 70 | 95/95 | 123.436 | 40 | 0.0 |
| 2202000 | 84.9 | 82.38 | 32 | 95/95 | 120.799 | 40 | 0.0 |
