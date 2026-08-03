# b10b-disc995seed2

![b10b-disc995seed2 progress](b10b-disc995seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 786000, avg score 94.1, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | b10b-disc995seed2 |
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

787 evals so far. Full series in [`b10b-disc995seed2_evals.json`](b10b-disc995seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.7 | 0.7 | 0 | 3/95 | -4.301 | 0 | 0.4 |
| 1000 | 1.1 | 1.1 | 0 | 4/95 | 0.083 | 0 | 0.4 |
| 2000 | 7.6 | 4.35 | 0 | 24/95 | 6.516 | 0 | 0.2 |
| ... | | | | | | | |
| 775000 | 93.5 | 92.32 | 91 | 95/95 | 142.107 | 50 | 0.0 |
| 776000 | 92.2 | 93.16 | 83 | 95/95 | 150.84 | 60 | 0.0 |
| 777000 | 91.1 | 92.86 | 83 | 95/95 | 129.676 | 40 | 0.0 |
| 778000 | 92.5 | 92.86 | 87 | 95/95 | 121.157 | 30 | 0.0 |
| 779000 | 94.6 | 92.78 | 93 | 95/95 | 173.139 | 80 | 0.0 |
| 780000 | 89.5 | 91.98 | 55 | 95/95 | 138.143 | 50 | 0.0 |
| 781000 | 94.1 | 92.36 | 90 | 95/95 | 172.576 | 80 | 0.0 |
| 782000 | 92.4 | 92.62 | 79 | 95/95 | 160.969 | 70 | 0.0 |
| 783000 | 90.2 | 92.16 | 53 | 95/95 | 158.682 | 70 | 0.0 |
| 784000 | 94.3 | 92.1 | 91 | 95/95 | 172.851 | 80 | 0.0 |
| 785000 | 90.0 | 92.2 | 66 | 95/95 | 118.206 | 30 | 0.0 |
| 786000 | 94.1 | 92.2 | 92 | 95/95 | 152.605 | 60 | 0.0 |
