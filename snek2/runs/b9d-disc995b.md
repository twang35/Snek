# b9d-disc995b

![b9d-disc995b progress](b9d-disc995b.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3447000, avg score 51.3, perfect games 0%.

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

3448 evals so far. Full series in [`b9d-disc995b_evals.json`](b9d-disc995b_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 6.1 | 6.1 | 0 | 12/95 | 1.084 | 0 | 0.4 |
| 2000 | 0.2 | 3.15 | 0 | 1/95 | -4.806 | 0 | 0.4 |
| ... | | | | | | | |
| 3436000 | 55.6 | 19.5 | 3 | 73/95 | 50.095 | 0 | 0.0 |
| 3437000 | 57.6 | 29.06 | 4 | 85/95 | 52.131 | 0 | 0.0 |
| 3438000 | 58.8 | 38.76 | 34 | 73/95 | 53.17 | 0 | 0.0 |
| 3439000 | 51.3 | 47.6 | 9 | 72/95 | 44.78 | 0 | 0.0 |
| 3440000 | 24.4 | 49.54 | 5 | 72/95 | 18.738 | 0 | 0.0 |
| 3441000 | 22.0 | 42.82 | 6 | 57/95 | 16.261 | 0 | 0.0 |
| 3442000 | 33.7 | 38.04 | 8 | 74/95 | 27.917 | 0 | 0.0 |
| 3443000 | 35.7 | 33.42 | 7 | 83/95 | 29.807 | 0 | 0.0 |
| 3444000 | 35.2 | 30.2 | 8 | 72/95 | 29.44 | 0 | 0.0 |
| 3445000 | 36.1 | 32.54 | 4 | 65/95 | 30.273 | 0 | 0.0 |
| 3446000 | 44.8 | 37.1 | 6 | 73/95 | 39.101 | 0 | 0.0 |
| 3447000 | 51.3 | 40.62 | 8 | 78/95 | 45.605 | 0 | 0.0 |
