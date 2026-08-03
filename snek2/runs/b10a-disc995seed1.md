# b10a-disc995seed1

![b10a-disc995seed1 progress](b10a-disc995seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1238000, avg score 86.2, perfect games 20%.

## Config

| setting | value |
|---|---|
| policy_name | b10a-disc995seed1 |
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

1239 evals so far. Full series in [`b10a-disc995seed1_evals.json`](b10a-disc995seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.901 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | -2.531 | 0 | 0.4 |
| 2000 | 0.7 | 0.7 | 0 | 2/95 | 0.142 | 0 | 0.4 |
| ... | | | | | | | |
| 1227000 | 88.4 | 90.8 | 70 | 95/95 | 116.021 | 30 | 0.0 |
| 1228000 | 79.9 | 88.72 | 23 | 95/95 | 108.128 | 30 | 0.0 |
| 1229000 | 79.5 | 86.26 | 41 | 95/95 | 107.75 | 30 | 0.0 |
| 1230000 | 87.0 | 85.32 | 45 | 95/95 | 145.384 | 60 | 0.0 |
| 1231000 | 90.8 | 85.12 | 83 | 95/95 | 119.095 | 30 | 0.0 |
| 1232000 | 85.8 | 84.6 | 68 | 95/95 | 123.465 | 40 | 0.0 |
| 1233000 | 83.7 | 85.36 | 63 | 92/95 | 82.294 | 0 | 0.0 |
| 1234000 | 87.7 | 87.0 | 83 | 95/95 | 105.703 | 20 | 0.0 |
| 1235000 | 83.8 | 86.36 | 35 | 95/95 | 132.312 | 50 | 0.0 |
| 1236000 | 92.7 | 86.74 | 78 | 95/95 | 160.695 | 70 | 0.0 |
| 1237000 | 88.2 | 87.22 | 80 | 95/95 | 116.3 | 30 | 0.0 |
| 1238000 | 86.2 | 87.72 | 72 | 95/95 | 104.653 | 20 | 0.0 |
