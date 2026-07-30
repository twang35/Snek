# b8g-clipseed3

![b8g-clipseed3 progress](b8g-clipseed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 4000, avg score 0.8, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b8g-clipseed3 |
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

5 evals so far. Full series in [`b8g-clipseed3_evals.json`](b8g-clipseed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 1000 | 6.4 | 6.4 | 0 | 9/95 | 4.928 | 0 | 0.4 |
| 2000 | 3.6 | 5.0 | 0 | 7/95 | -1.41 | 0 | 0.4 |
| 3000 | 0.4 | 3.47 | 0 | 2/95 | -1.488 | 0 | 0.4 |
| 4000 | 0.8 | 2.8 | 0 | 3/95 | 0.243 | 0 | 0.4 |
