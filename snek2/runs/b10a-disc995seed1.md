# b10a-disc995seed1

![b10a-disc995seed1 progress](b10a-disc995seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 750000, avg score 85.1, perfect games 60%.

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

751 evals so far. Full series in [`b10a-disc995seed1_evals.json`](b10a-disc995seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.901 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | -2.531 | 0 | 0.4 |
| 2000 | 0.7 | 0.7 | 0 | 2/95 | 0.142 | 0 | 0.4 |
| ... | | | | | | | |
| 739000 | 93.1 | 91.3 | 79 | 95/95 | 171.643 | 80 | 0.0 |
| 740000 | 91.4 | 90.84 | 80 | 95/95 | 129.78 | 40 | 0.0 |
| 741000 | 91.9 | 90.84 | 85 | 95/95 | 140.216 | 50 | 0.0 |
| 742000 | 91.2 | 92.14 | 75 | 95/95 | 169.787 | 80 | 0.0 |
| 743000 | 86.6 | 90.84 | 11 | 95/95 | 175.198 | 90 | 0.0 |
| 744000 | 93.8 | 90.98 | 90 | 95/95 | 162.296 | 70 | 0.0 |
| 745000 | 69.6 | 86.62 | 9 | 95/95 | 138.313 | 70 | 0.0 |
| 746000 | 94.7 | 87.18 | 92 | 95/95 | 183.147 | 90 | 0.0 |
| 747000 | 88.8 | 86.7 | 40 | 95/95 | 166.842 | 80 | 0.0 |
| 748000 | 93.2 | 88.02 | 83 | 95/95 | 161.7 | 70 | 0.0 |
| 749000 | 91.9 | 87.64 | 82 | 95/95 | 150.38 | 60 | 0.0 |
| 750000 | 85.1 | 90.74 | 11 | 95/95 | 143.743 | 60 | 0.0 |
