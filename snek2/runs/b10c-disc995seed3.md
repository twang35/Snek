# b10c-disc995seed3

![b10c-disc995seed3 progress](b10c-disc995seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1949000, avg score 89.6, perfect games 30%.

## Config

| setting | value |
|---|---|
| policy_name | b10c-disc995seed3 |
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

1950 evals so far. Full series in [`b10c-disc995seed3_evals.json`](b10c-disc995seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 1.9 | 1.9 | 0 | 5/95 | 0.89 | 0 | 0.4 |
| 2000 | 2.7 | 2.3 | 0 | 8/95 | 1.691 | 0 | 0.4 |
| ... | | | | | | | |
| 1938000 | 89.2 | 88.26 | 81 | 95/95 | 107.603 | 20 | 0.0 |
| 1939000 | 81.9 | 87.86 | 1 | 95/95 | 110.429 | 30 | 0.0 |
| 1940000 | 88.7 | 87.8 | 79 | 95/95 | 107.17 | 20 | 0.0 |
| 1941000 | 91.0 | 87.56 | 77 | 95/95 | 119.378 | 30 | 0.0 |
| 1942000 | 87.0 | 87.56 | 71 | 95/95 | 95.194 | 10 | 0.0 |
| 1943000 | 92.1 | 88.14 | 83 | 95/95 | 139.992 | 50 | 0.0 |
| 1944000 | 92.4 | 90.24 | 83 | 95/95 | 140.471 | 50 | 0.0 |
| 1945000 | 83.7 | 89.24 | 41 | 95/95 | 102.042 | 20 | 0.0 |
| 1946000 | 94.3 | 89.9 | 92 | 95/95 | 162.719 | 70 | 0.0 |
| 1947000 | 90.0 | 90.5 | 85 | 95/95 | 108.028 | 20 | 0.0 |
| 1948000 | 87.6 | 89.6 | 73 | 95/95 | 116.146 | 30 | 0.0 |
| 1949000 | 89.6 | 89.04 | 67 | 95/95 | 118.142 | 30 | 0.0 |
