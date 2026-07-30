# b8b-disc999seed2

![b8b-disc999seed2 progress](b8b-disc999seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1421000, avg score 0.1, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b8b-disc999seed2 |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.999 |
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

1422 evals so far. Full series in [`b8b-disc999seed2_evals.json`](b8b-disc999seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.2 | 0.2 | 0 | 1/95 | -3.024 | 0 | 0.4 |
| 2000 | 0.5 | 0.35 | 0 | 2/95 | -1.398 | 0 | 0.4 |
| ... | | | | | | | |
| 1410000 | 0.0 | 0.02 | 0 | 0/95 | -5.001 | 0 | 0.01 |
| 1411000 | 0.1 | 0.04 | 0 | 1/95 | -4.902 | 0 | 0.01 |
| 1412000 | 0.0 | 0.04 | 0 | 0/95 | -5.002 | 0 | 0.01 |
| 1413000 | 0.0 | 0.02 | 0 | 0/95 | -5.001 | 0 | 0.01 |
| 1414000 | 0.0 | 0.02 | 0 | 0/95 | -5.001 | 0 | 0.01 |
| 1415000 | 0.0 | 0.02 | 0 | 0/95 | -5.002 | 0 | 0.01 |
| 1416000 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.01 |
| 1417000 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.01 |
| 1418000 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.01 |
| 1419000 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.01 |
| 1420000 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.01 |
| 1421000 | 0.1 | 0.02 | 0 | 1/95 | -4.902 | 0 | 0.01 |
