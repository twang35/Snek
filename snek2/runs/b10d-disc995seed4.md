# b10d-disc995seed4

![b10d-disc995seed4 progress](b10d-disc995seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 721000, avg score 93.0, perfect games 60%.

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

722 evals so far. Full series in [`b10d-disc995seed4_evals.json`](b10d-disc995seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.004 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 2/95 | -0.152 | 0 | 0.4 |
| 2000 | 0.0 | 0.2 | 0 | 0/95 | -0.548 | 0 | 0.4 |
| ... | | | | | | | |
| 710000 | 92.5 | 90.54 | 85 | 95/95 | 160.497 | 70 | 0.0 |
| 711000 | 90.8 | 90.52 | 79 | 95/95 | 138.872 | 50 | 0.0 |
| 712000 | 89.8 | 89.78 | 53 | 95/95 | 168.277 | 80 | 0.0 |
| 713000 | 93.2 | 90.1 | 87 | 95/95 | 161.61 | 70 | 0.0 |
| 714000 | 90.9 | 91.44 | 79 | 95/95 | 139.414 | 50 | 0.0 |
| 715000 | 88.4 | 90.62 | 35 | 95/95 | 156.809 | 70 | 0.0 |
| 716000 | 91.4 | 90.74 | 81 | 95/95 | 149.851 | 60 | 0.0 |
| 717000 | 87.8 | 90.34 | 51 | 95/95 | 136.264 | 50 | 0.0 |
| 718000 | 94.6 | 90.62 | 91 | 95/95 | 182.923 | 90 | 0.0 |
| 719000 | 92.6 | 90.96 | 85 | 95/95 | 151.017 | 60 | 0.0 |
| 720000 | 90.5 | 91.38 | 76 | 95/95 | 128.525 | 40 | 0.0 |
| 721000 | 93.0 | 91.7 | 82 | 95/95 | 150.583 | 60 | 0.0 |
