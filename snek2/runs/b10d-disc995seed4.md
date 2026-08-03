# b10d-disc995seed4

![b10d-disc995seed4 progress](b10d-disc995seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 29000, avg score 72.0, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b10d-disc995seed4 |
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

30 evals so far. Full series in [`b10d-disc995seed4_evals.json`](b10d-disc995seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.004 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 2/95 | -0.152 | 0 | 0.4 |
| 2000 | 0.0 | 0.2 | 0 | 0/95 | -0.548 | 0 | 0.4 |
| ... | | | | | | | |
| 18000 | 58.3 | 60.0 | 2 | 84/95 | 57.117 | 0 | 0.001 |
| 19000 | 67.7 | 62.02 | 13 | 81/95 | 66.154 | 0 | 0.001 |
| 20000 | 63.7 | 61.96 | 8 | 83/95 | 61.918 | 0 | 0.001 |
| 21000 | 34.8 | 56.56 | 0 | 82/95 | 33.799 | 0 | 0.001 |
| 22000 | 23.6 | 49.62 | 0 | 77/95 | 22.766 | 0 | 0.001 |
| 23000 | 64.5 | 50.86 | 2 | 85/95 | 62.185 | 0 | 0.001 |
| 24000 | 73.3 | 51.98 | 17 | 86/95 | 71.409 | 0 | 0.001 |
| 25000 | 35.7 | 46.38 | 0 | 84/95 | 34.66 | 0 | 0.001 |
| 26000 | 77.9 | 55.0 | 60 | 87/95 | 75.888 | 0 | 0.001 |
| 27000 | 78.8 | 66.04 | 69 | 86/95 | 77.151 | 0 | 0.001 |
| 28000 | 66.3 | 66.4 | 0 | 89/95 | 64.886 | 0 | 0.001 |
| 29000 | 72.0 | 66.14 | 0 | 87/95 | 70.542 | 0 | 0.001 |
