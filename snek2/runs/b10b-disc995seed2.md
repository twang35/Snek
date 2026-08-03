# b10b-disc995seed2

![b10b-disc995seed2 progress](b10b-disc995seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1334000, avg score 86.7, perfect games 60%.

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

1335 evals so far. Full series in [`b10b-disc995seed2_evals.json`](b10b-disc995seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.7 | 0.7 | 0 | 3/95 | -4.301 | 0 | 0.4 |
| 1000 | 1.1 | 1.1 | 0 | 4/95 | 0.083 | 0 | 0.4 |
| 2000 | 7.6 | 4.35 | 0 | 24/95 | 6.516 | 0 | 0.2 |
| ... | | | | | | | |
| 1323000 | 94.0 | 91.76 | 85 | 95/95 | 182.025 | 90 | 0.0 |
| 1324000 | 87.2 | 90.64 | 27 | 95/95 | 155.817 | 70 | 0.0 |
| 1325000 | 93.0 | 90.42 | 75 | 95/95 | 181.466 | 90 | 0.0 |
| 1326000 | 91.7 | 89.82 | 71 | 95/95 | 149.836 | 60 | 0.0 |
| 1327000 | 93.9 | 91.96 | 87 | 95/95 | 172.375 | 80 | 0.0 |
| 1328000 | 93.2 | 91.8 | 84 | 95/95 | 161.207 | 70 | 0.0 |
| 1329000 | 92.2 | 92.8 | 83 | 95/95 | 160.641 | 70 | 0.0 |
| 1330000 | 93.1 | 92.82 | 79 | 95/95 | 171.528 | 80 | 0.0 |
| 1331000 | 93.4 | 93.16 | 89 | 95/95 | 151.803 | 60 | 0.0 |
| 1332000 | 90.2 | 92.42 | 76 | 95/95 | 137.774 | 50 | 0.0 |
| 1333000 | 92.8 | 92.34 | 81 | 95/95 | 161.262 | 70 | 0.0 |
| 1334000 | 86.7 | 91.24 | 29 | 95/95 | 144.334 | 60 | 0.0 |
