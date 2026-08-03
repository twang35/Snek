# b10c-disc995seed3

![b10c-disc995seed3 progress](b10c-disc995seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 4122000, avg score 93.9, perfect games 80%.

## Config

| setting | value |
|---|---|
| policy_name | b10c-disc995seed3 |
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

4123 evals so far. Full series in [`b10c-disc995seed3_evals.json`](b10c-disc995seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 1.9 | 1.9 | 0 | 5/95 | 0.89 | 0 | 0.4 |
| 2000 | 2.7 | 2.3 | 0 | 8/95 | 1.691 | 0 | 0.4 |
| ... | | | | | | | |
| 4111000 | 87.1 | 90.98 | 83 | 95/95 | 95.398 | 10 | 0.0 |
| 4112000 | 89.4 | 90.06 | 75 | 95/95 | 127.352 | 40 | 0.0 |
| 4113000 | 92.0 | 89.9 | 80 | 95/95 | 150.316 | 60 | 0.0 |
| 4114000 | 92.0 | 90.1 | 79 | 95/95 | 170.387 | 80 | 0.0 |
| 4115000 | 87.2 | 89.54 | 48 | 95/95 | 135.291 | 50 | 0.0 |
| 4116000 | 92.1 | 90.54 | 82 | 95/95 | 150.471 | 60 | 0.0 |
| 4117000 | 93.6 | 91.38 | 87 | 95/95 | 172.018 | 80 | 0.0 |
| 4118000 | 91.5 | 91.28 | 82 | 95/95 | 139.831 | 50 | 0.0 |
| 4119000 | 89.2 | 90.72 | 73 | 95/95 | 137.209 | 50 | 0.0 |
| 4120000 | 88.3 | 90.94 | 58 | 95/95 | 155.846 | 70 | 0.0 |
| 4121000 | 93.0 | 91.12 | 89 | 95/95 | 141.301 | 50 | 0.0 |
| 4122000 | 93.9 | 91.18 | 87 | 95/95 | 171.694 | 80 | 0.0 |
