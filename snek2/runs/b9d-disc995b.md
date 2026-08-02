# b9d-disc995b

![b9d-disc995b progress](b9d-disc995b.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3367000, avg score 39.9, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b9d-disc995b |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.995 |
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
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

3368 evals so far. Full series in [`b9d-disc995b_evals.json`](b9d-disc995b_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 6.1 | 6.1 | 0 | 12/95 | 1.084 | 0 | 0.4 |
| 2000 | 0.2 | 3.15 | 0 | 1/95 | -4.806 | 0 | 0.4 |
| ... | | | | | | | |
| 3356000 | 44.6 | 43.88 | 4 | 83/95 | 38.955 | 0 | 0.0 |
| 3357000 | 53.0 | 43.88 | 3 | 80/95 | 47.048 | 0 | 0.0 |
| 3358000 | 48.7 | 41.86 | 4 | 81/95 | 42.941 | 0 | 0.0 |
| 3359000 | 56.1 | 47.68 | 7 | 88/95 | 50.242 | 0 | 0.0 |
| 3360000 | 44.8 | 49.44 | 7 | 76/95 | 39.163 | 0 | 0.0 |
| 3361000 | 51.4 | 50.8 | 8 | 80/95 | 45.288 | 0 | 0.0 |
| 3362000 | 30.5 | 46.3 | 2 | 66/95 | 24.907 | 0 | 0.0 |
| 3363000 | 33.7 | 43.3 | 2 | 78/95 | 28.05 | 0 | 0.0 |
| 3364000 | 47.1 | 41.5 | 3 | 77/95 | 41.339 | 0 | 0.0 |
| 3365000 | 63.5 | 45.24 | 7 | 78/95 | 57.608 | 0 | 0.0 |
| 3366000 | 63.9 | 47.74 | 32 | 86/95 | 58.07 | 0 | 0.0 |
| 3367000 | 39.9 | 49.62 | 4 | 77/95 | 33.968 | 0 | 0.0 |
