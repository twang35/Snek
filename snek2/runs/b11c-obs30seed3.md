# b11c-obs30seed3

![b11c-obs30seed3 progress](b11c-obs30seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3232000, avg score 93.4, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | b11c-obs30seed3 |
| seed | 3 |
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

3233 evals so far. Full series in [`b11c-obs30seed3_evals.json`](b11c-obs30seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.902 | 0 | 0.4 |
| 1000 | 1.0 | 1.0 | 0 | 4/95 | 0.445 | 0 | 0.4 |
| 2000 | 0.9 | 0.95 | 0 | 4/95 | 0.347 | 0 | 0.4 |
| ... | | | | | | | |
| 3221000 | 90.0 | 73.26 | 70 | 95/95 | 148.499 | 60 | 0.0 |
| 3222000 | 83.6 | 71.42 | 3 | 95/95 | 141.984 | 60 | 0.0 |
| 3223000 | 93.7 | 85.24 | 92 | 95/95 | 120.328 | 30 | 0.0 |
| 3224000 | 94.5 | 85.54 | 90 | 95/95 | 182.706 | 90 | 0.0 |
| 3225000 | 92.7 | 90.9 | 78 | 95/95 | 161.062 | 70 | 0.0 |
| 3226000 | 87.3 | 90.36 | 26 | 95/95 | 113.741 | 30 | 0.0 |
| 3227000 | 91.1 | 91.86 | 81 | 95/95 | 98.352 | 10 | 0.0 |
| 3228000 | 91.6 | 91.44 | 86 | 95/95 | 139.624 | 50 | 0.0 |
| 3229000 | 93.4 | 91.22 | 88 | 95/95 | 140.95 | 50 | 0.0 |
| 3230000 | 84.8 | 89.64 | 4 | 95/95 | 121.66 | 40 | 0.0 |
| 3231000 | 94.3 | 91.04 | 92 | 95/95 | 140.971 | 50 | 0.0 |
| 3232000 | 93.4 | 91.5 | 84 | 95/95 | 150.912 | 60 | 0.0 |
