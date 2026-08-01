# b8d-disc995clip

![b8d-disc995clip progress](b8d-disc995clip.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 11643000, avg score 0.6, perfect games 0%.

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

11644 evals so far. Full series in [`b8d-disc995clip_evals.json`](b8d-disc995clip_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.5 | 0.5 | 0 | 2/95 | -3.652 | 0 | 0.4 |
| 2000 | 0.1 | 0.3 | 0 | 1/95 | -4.93 | 0 | 0.4 |
| ... | | | | | | | |
| 11632000 | 1.1 | 0.62 | 0 | 5/95 | -3.903 | 0 | 0.0 |
| 11633000 | 1.5 | 0.88 | 0 | 6/95 | -3.503 | 0 | 0.0 |
| 11634000 | 1.3 | 1.06 | 0 | 6/95 | -3.704 | 0 | 0.0 |
| 11635000 | 1.3 | 1.2 | 0 | 4/95 | -3.703 | 0 | 0.0 |
| 11636000 | 1.9 | 1.42 | 0 | 4/95 | -3.104 | 0 | 0.0 |
| 11637000 | 0.9 | 1.38 | 0 | 4/95 | -4.103 | 0 | 0.0 |
| 11638000 | 0.2 | 1.12 | 0 | 2/95 | -4.802 | 0 | 0.0 |
| 11639000 | 1.9 | 1.24 | 0 | 7/95 | -3.104 | 0 | 0.0 |
| 11640000 | 1.0 | 1.18 | 0 | 4/95 | -4.003 | 0 | 0.0 |
| 11641000 | 1.3 | 1.06 | 0 | 6/95 | -3.703 | 0 | 0.0 |
| 11642000 | 1.4 | 1.16 | 0 | 6/95 | -3.603 | 0 | 0.0 |
| 11643000 | 0.6 | 1.24 | 0 | 4/95 | -4.402 | 0 | 0.0 |
