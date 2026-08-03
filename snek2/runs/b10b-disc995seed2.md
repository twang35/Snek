# b10b-disc995seed2

![b10b-disc995seed2 progress](b10b-disc995seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 899000, avg score 91.7, perfect games 60%.

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

900 evals so far. Full series in [`b10b-disc995seed2_evals.json`](b10b-disc995seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.7 | 0.7 | 0 | 3/95 | -4.301 | 0 | 0.4 |
| 1000 | 1.1 | 1.1 | 0 | 4/95 | 0.083 | 0 | 0.4 |
| 2000 | 7.6 | 4.35 | 0 | 24/95 | 6.516 | 0 | 0.2 |
| ... | | | | | | | |
| 888000 | 85.5 | 91.7 | 26 | 95/95 | 132.427 | 50 | 0.0 |
| 889000 | 92.8 | 91.54 | 90 | 95/95 | 120.561 | 30 | 0.0 |
| 890000 | 94.2 | 91.62 | 91 | 95/95 | 162.771 | 70 | 0.0 |
| 891000 | 94.4 | 91.9 | 91 | 95/95 | 172.401 | 80 | 0.0 |
| 892000 | 94.8 | 92.34 | 93 | 95/95 | 183.241 | 90 | 0.0 |
| 893000 | 92.1 | 93.66 | 79 | 95/95 | 129.876 | 40 | 0.0 |
| 894000 | 94.1 | 93.92 | 86 | 95/95 | 182.553 | 90 | 0.0 |
| 895000 | 94.2 | 93.92 | 87 | 95/95 | 182.713 | 90 | 0.0 |
| 896000 | 90.5 | 93.14 | 57 | 95/95 | 169.158 | 80 | 0.0 |
| 897000 | 93.5 | 92.88 | 85 | 95/95 | 172.018 | 80 | 0.0 |
| 898000 | 89.6 | 92.38 | 41 | 95/95 | 178.129 | 90 | 0.0 |
| 899000 | 91.7 | 91.9 | 79 | 95/95 | 150.277 | 60 | 0.0 |
