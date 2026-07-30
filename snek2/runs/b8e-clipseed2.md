# b8e-clipseed2

![b8e-clipseed2 progress](b8e-clipseed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 16000, avg score 8.6, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b8e-clipseed2 |
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

17 evals so far. Full series in [`b8e-clipseed2_evals.json`](b8e-clipseed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.2 | 0.2 | 0 | 1/95 | -4.839 | 0 | 0.4 |
| 2000 | 2.4 | 1.3 | 0 | 8/95 | -2.67 | 0 | 0.4 |
| ... | | | | | | | |
| 5000 | 0.0 | 3.54 | 0 | 0/95 | -5.006 | 0 | 0.2 |
| 6000 | 3.6 | 4.22 | 1 | 9/95 | -1.426 | 0 | 0.2 |
| 7000 | 2.3 | 4.2 | 0 | 5/95 | -2.732 | 0 | 0.2 |
| 8000 | 3.2 | 2.88 | 0 | 10/95 | -1.847 | 0 | 0.2 |
| 9000 | 0.1 | 1.84 | 0 | 1/95 | -4.949 | 0 | 0.2 |
| 10000 | 0.4 | 1.92 | 0 | 1/95 | -3.754 | 0 | 0.2 |
| 11000 | 4.2 | 2.04 | 1 | 9/95 | 0.072 | 0 | 0.2 |
| 12000 | 3.5 | 2.28 | 0 | 9/95 | 0.271 | 0 | 0.2 |
| 13000 | 2.4 | 2.12 | 0 | 8/95 | 0.505 | 0 | 0.2 |
| 14000 | 4.3 | 2.96 | 1 | 10/95 | 1.072 | 0 | 0.2 |
| 15000 | 4.0 | 3.68 | 0 | 11/95 | 1.218 | 0 | 0.2 |
| 16000 | 8.6 | 4.56 | 2 | 16/95 | 4.424 | 0 | 0.2 |
