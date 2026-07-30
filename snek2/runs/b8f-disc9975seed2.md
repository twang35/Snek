# b8f-disc9975seed2

![b8f-disc9975seed2 progress](b8f-disc9975seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 19000, avg score 4.7, perfect games 0%.

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

20 evals so far. Full series in [`b8f-disc9975seed2_evals.json`](b8f-disc9975seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.3 | 0.3 | 0 | 1/95 | -4.703 | 0 | 0.4 |
| 1000 | 11.3 | 11.3 | 0 | 17/95 | 6.262 | 0 | 0.2 |
| 2000 | 0.1 | 5.7 | 0 | 1/95 | -4.906 | 0 | 0.2 |
| ... | | | | | | | |
| 8000 | 0.0 | 0.46 | 0 | 0/95 | -5.002 | 0 | 0.2 |
| 9000 | 0.2 | 0.44 | 0 | 1/95 | -4.812 | 0 | 0.2 |
| 10000 | 4.4 | 0.92 | 2 | 8/95 | -0.61 | 0 | 0.2 |
| 11000 | 0.2 | 0.96 | 0 | 1/95 | -4.813 | 0 | 0.2 |
| 12000 | 7.6 | 2.48 | 4 | 14/95 | 2.58 | 0 | 0.2 |
| 13000 | 7.1 | 3.9 | 4 | 12/95 | 2.088 | 0 | 0.2 |
| 14000 | 5.8 | 5.02 | 2 | 9/95 | 1.231 | 0 | 0.2 |
| 15000 | 4.5 | 5.04 | 2 | 8/95 | -0.509 | 0 | 0.2 |
| 16000 | 7.0 | 6.4 | 3 | 12/95 | 1.984 | 0 | 0.2 |
| 17000 | 4.9 | 5.86 | 2 | 10/95 | -0.112 | 0 | 0.2 |
| 18000 | 3.8 | 5.2 | 2 | 7/95 | -1.247 | 0 | 0.2 |
| 19000 | 4.7 | 4.98 | 2 | 9/95 | -0.311 | 0 | 0.2 |
