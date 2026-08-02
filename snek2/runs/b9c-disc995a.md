# b9c-disc995a

![b9c-disc995a progress](b9c-disc995a.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3568000, avg score 85.0, perfect games 10%.

## Config

| setting | value |
|---|---|
| policy_name | b9c-disc995a |
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

3569 evals so far. Full series in [`b9c-disc995a_evals.json`](b9c-disc995a_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -5.023 | 0 | 0.4 |
| 2000 | 0.2 | 0.1 | 0 | 1/95 | -3.022 | 0 | 0.4 |
| ... | | | | | | | |
| 3557000 | 88.9 | 79.48 | 76 | 95/95 | 114.546 | 30 | 0.0 |
| 3558000 | 71.4 | 77.0 | 36 | 95/95 | 86.597 | 20 | 0.0 |
| 3559000 | 80.4 | 77.68 | 40 | 95/95 | 95.664 | 20 | 0.0 |
| 3560000 | 79.9 | 79.86 | 4 | 95/95 | 115.968 | 40 | 0.0 |
| 3561000 | 78.1 | 79.74 | 40 | 95/95 | 93.145 | 20 | 0.0 |
| 3562000 | 73.8 | 76.72 | 40 | 95/95 | 89.169 | 20 | 0.0 |
| 3563000 | 84.0 | 79.24 | 64 | 95/95 | 130.487 | 50 | 0.0 |
| 3564000 | 77.7 | 78.7 | 33 | 95/95 | 92.915 | 20 | 0.0 |
| 3565000 | 76.9 | 78.1 | 6 | 95/95 | 123.465 | 50 | 0.0 |
| 3566000 | 80.7 | 78.62 | 39 | 95/95 | 106.338 | 30 | 0.0 |
| 3567000 | 80.7 | 80.0 | 29 | 95/95 | 116.76 | 40 | 0.0 |
| 3568000 | 85.0 | 80.2 | 72 | 95/95 | 89.803 | 10 | 0.0 |
