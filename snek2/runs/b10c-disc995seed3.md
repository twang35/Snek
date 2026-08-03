# b10c-disc995seed3

![b10c-disc995seed3 progress](b10c-disc995seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 764000, avg score 82.1, perfect games 0%.

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

765 evals so far. Full series in [`b10c-disc995seed3_evals.json`](b10c-disc995seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 1.9 | 1.9 | 0 | 5/95 | 0.89 | 0 | 0.4 |
| 2000 | 2.7 | 2.3 | 0 | 8/95 | 1.691 | 0 | 0.4 |
| ... | | | | | | | |
| 753000 | 81.6 | 81.1 | 53 | 95/95 | 89.338 | 10 | 0.0 |
| 754000 | 84.7 | 82.18 | 58 | 95/95 | 123.216 | 40 | 0.0 |
| 755000 | 85.8 | 84.08 | 71 | 95/95 | 114.276 | 30 | 0.0 |
| 756000 | 84.2 | 84.46 | 67 | 95/95 | 92.25 | 10 | 0.0 |
| 757000 | 80.3 | 83.32 | 62 | 95/95 | 88.846 | 10 | 0.0 |
| 758000 | 85.4 | 84.08 | 73 | 95/95 | 93.082 | 10 | 0.0 |
| 759000 | 81.4 | 83.42 | 67 | 92/95 | 79.212 | 0 | 0.0 |
| 760000 | 85.2 | 83.3 | 61 | 95/95 | 122.811 | 40 | 0.0 |
| 761000 | 84.9 | 83.44 | 68 | 95/95 | 103.03 | 20 | 0.0 |
| 762000 | 82.4 | 83.86 | 62 | 95/95 | 101.064 | 20 | 0.0 |
| 763000 | 86.5 | 84.08 | 75 | 95/95 | 104.604 | 20 | 0.0 |
| 764000 | 82.1 | 84.22 | 71 | 92/95 | 79.368 | 0 | 0.0 |
