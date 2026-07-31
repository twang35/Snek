# b8f-disc9975seed2

![b8f-disc9975seed2 progress](b8f-disc9975seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3045000, avg score 92.1, perfect games 70%.

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

3046 evals so far. Full series in [`b8f-disc9975seed2_evals.json`](b8f-disc9975seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.3 | 0.3 | 0 | 1/95 | -4.703 | 0 | 0.4 |
| 1000 | 11.3 | 11.3 | 0 | 17/95 | 6.262 | 0 | 0.2 |
| 2000 | 0.1 | 5.7 | 0 | 1/95 | -4.906 | 0 | 0.2 |
| ... | | | | | | | |
| 3034000 | 83.9 | 83.28 | 3 | 95/95 | 160.801 | 80 | 0.0 |
| 3035000 | 65.0 | 77.82 | 0 | 95/95 | 90.125 | 30 | 0.0 |
| 3036000 | 93.9 | 78.72 | 84 | 95/95 | 180.801 | 90 | 0.0 |
| 3037000 | 89.2 | 82.12 | 76 | 95/95 | 145.059 | 60 | 0.0 |
| 3038000 | 74.3 | 81.26 | 1 | 95/95 | 120.007 | 50 | 0.0 |
| 3039000 | 87.8 | 82.04 | 60 | 95/95 | 143.731 | 60 | 0.0 |
| 3040000 | 78.3 | 84.7 | 6 | 95/95 | 124.092 | 50 | 0.0 |
| 3041000 | 88.3 | 83.58 | 54 | 95/95 | 154.717 | 70 | 0.0 |
| 3042000 | 80.7 | 81.88 | 56 | 95/95 | 95.345 | 20 | 0.0 |
| 3043000 | 89.6 | 84.94 | 56 | 95/95 | 156.003 | 70 | 0.0 |
| 3044000 | 89.0 | 85.18 | 52 | 95/95 | 155.153 | 70 | 0.0 |
| 3045000 | 92.1 | 87.94 | 82 | 95/95 | 158.513 | 70 | 0.0 |
