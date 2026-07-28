# b5b-schlong2

![b5b-schlong2 progress](b5b-schlong2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 44000, avg score 52.2, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b5b-schlong2 |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.99 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
| initial_epsilon | 0.4 |
| min_epsilon | 0.0 |
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.8 |
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

## Evals

45 evals so far. Full series in [`b5b-schlong2_evals.json`](b5b-schlong2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.004 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -5.008 | 0 | 0.4 |
| 2000 | 0.0 | 0.0 | 0 | 0/95 | -5.007 | 0 | 0.4 |
| ... | | | | | | | |
| 33000 | 11.6 | 9.0 | 6 | 29/95 | 6.583 | 0 | 0.2 |
| 34000 | 33.2 | 14.56 | 19 | 56/95 | 28.115 | 0 | 0.1 |
| 35000 | 61.3 | 25.52 | 36 | 80/95 | 55.917 | 0 | 0.05 |
| 36000 | 66.1 | 36.9 | 32 | 81/95 | 61.145 | 0 | 0.01 |
| 37000 | 52.6 | 44.96 | 17 | 79/95 | 48.144 | 0 | 0.01 |
| 38000 | 46.2 | 51.88 | 16 | 76/95 | 41.831 | 0 | 0.01 |
| 39000 | 44.0 | 54.04 | 22 | 68/95 | 38.671 | 0 | 0.01 |
| 40000 | 61.3 | 54.04 | 41 | 78/95 | 56.188 | 0 | 0.01 |
| 41000 | 59.2 | 52.66 | 11 | 78/95 | 54.164 | 0 | 0.01 |
| 42000 | 60.1 | 54.16 | 32 | 78/95 | 54.655 | 0 | 0.01 |
| 43000 | 55.6 | 56.04 | 24 | 85/95 | 50.508 | 0 | 0.01 |
| 44000 | 52.2 | 57.68 | 25 | 76/95 | 46.867 | 0 | 0.01 |
