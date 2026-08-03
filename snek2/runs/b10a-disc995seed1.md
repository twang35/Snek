# b10a-disc995seed1

![b10a-disc995seed1 progress](b10a-disc995seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 4289000, avg score 94.2, perfect games 90%.

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

4290 evals so far. Full series in [`b10a-disc995seed1_evals.json`](b10a-disc995seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.901 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | -2.531 | 0 | 0.4 |
| 2000 | 0.7 | 0.7 | 0 | 2/95 | 0.142 | 0 | 0.4 |
| ... | | | | | | | |
| 4278000 | 91.4 | 91.26 | 82 | 95/95 | 128.977 | 40 | 0.0 |
| 4279000 | 92.6 | 90.94 | 85 | 95/95 | 160.91 | 70 | 0.0 |
| 4280000 | 91.5 | 90.76 | 76 | 95/95 | 139.498 | 50 | 0.0 |
| 4281000 | 90.4 | 90.64 | 76 | 95/95 | 157.921 | 70 | 0.0 |
| 4282000 | 92.6 | 91.7 | 88 | 95/95 | 141.084 | 50 | 0.0 |
| 4283000 | 93.8 | 92.18 | 90 | 95/95 | 162.075 | 70 | 0.0 |
| 4284000 | 92.5 | 92.16 | 85 | 95/95 | 150.807 | 60 | 0.0 |
| 4285000 | 91.2 | 92.1 | 74 | 95/95 | 149.18 | 60 | 0.0 |
| 4286000 | 91.5 | 92.32 | 85 | 95/95 | 129.412 | 40 | 0.0 |
| 4287000 | 92.3 | 92.26 | 83 | 95/95 | 150.246 | 60 | 0.0 |
| 4288000 | 91.2 | 91.74 | 79 | 95/95 | 128.716 | 40 | 0.0 |
| 4289000 | 94.2 | 92.08 | 87 | 95/95 | 182.546 | 90 | 0.0 |
