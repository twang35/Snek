# b11b-obs30seed2

![b11b-obs30seed2 progress](b11b-obs30seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 19000, avg score 64.6, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b11b-obs30seed2 |
| seed | 2 |
| zeroed_observations | none |
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

20 evals so far. Full series in [`b11b-obs30seed2_evals.json`](b11b-obs30seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.003 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | -0.745 | 0 | 0.4 |
| 2000 | 0.7 | 0.7 | 0 | 2/95 | 0.146 | 0 | 0.4 |
| ... | | | | | | | |
| 8000 | 35.1 | 16.82 | 21 | 50/95 | 29.961 | 0 | 0.05 |
| 9000 | 30.9 | 22.54 | 10 | 60/95 | 25.782 | 0 | 0.05 |
| 10000 | 38.3 | 29.9 | 27 | 68/95 | 33.14 | 0 | 0.05 |
| 11000 | 39.0 | 34.32 | 12 | 73/95 | 33.825 | 0 | 0.05 |
| 12000 | 65.4 | 41.74 | 30 | 95/95 | 70.715 | 10 | 0.01 |
| 13000 | 66.1 | 47.94 | 29 | 92/95 | 62.356 | 0 | 0.001 |
| 14000 | 70.2 | 55.8 | 41 | 91/95 | 66.666 | 0 | 0.001 |
| 15000 | 73.8 | 62.9 | 60 | 85/95 | 72.198 | 0 | 0.001 |
| 16000 | 68.3 | 68.76 | 29 | 93/95 | 66.859 | 0 | 0.001 |
| 17000 | 77.1 | 71.1 | 49 | 87/95 | 75.587 | 0 | 0.001 |
| 18000 | 69.5 | 71.78 | 52 | 81/95 | 68.057 | 0 | 0.001 |
| 19000 | 64.6 | 70.66 | 39 | 79/95 | 63.192 | 0 | 0.001 |
