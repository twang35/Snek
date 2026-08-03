# b10b-disc995seed2

![b10b-disc995seed2 progress](b10b-disc995seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1258000, avg score 89.0, perfect games 70%.

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

1259 evals so far. Full series in [`b10b-disc995seed2_evals.json`](b10b-disc995seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.7 | 0.7 | 0 | 3/95 | -4.301 | 0 | 0.4 |
| 1000 | 1.1 | 1.1 | 0 | 4/95 | 0.083 | 0 | 0.4 |
| 2000 | 7.6 | 4.35 | 0 | 24/95 | 6.516 | 0 | 0.2 |
| ... | | | | | | | |
| 1247000 | 86.5 | 92.5 | 65 | 95/95 | 114.692 | 30 | 0.0 |
| 1248000 | 89.6 | 91.8 | 43 | 95/95 | 168.135 | 80 | 0.0 |
| 1249000 | 94.5 | 92.0 | 90 | 95/95 | 183.053 | 90 | 0.0 |
| 1250000 | 94.2 | 91.84 | 89 | 95/95 | 172.672 | 80 | 0.0 |
| 1251000 | 95.0 | 91.96 | 95 | 95/95 | 193.486 | 100 | 0.0 |
| 1252000 | 91.8 | 93.02 | 81 | 95/95 | 139.501 | 50 | 0.0 |
| 1253000 | 94.5 | 94.0 | 91 | 95/95 | 172.615 | 80 | 0.0 |
| 1254000 | 86.9 | 92.48 | 21 | 95/95 | 144.659 | 60 | 0.0 |
| 1255000 | 93.7 | 92.38 | 87 | 95/95 | 172.27 | 80 | 0.0 |
| 1256000 | 92.0 | 91.78 | 65 | 95/95 | 180.506 | 90 | 0.0 |
| 1257000 | 85.4 | 90.5 | 13 | 95/95 | 153.556 | 70 | 0.0 |
| 1258000 | 89.0 | 89.4 | 47 | 95/95 | 157.609 | 70 | 0.0 |
