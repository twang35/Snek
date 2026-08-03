# b10c-disc995seed3

![b10c-disc995seed3 progress](b10c-disc995seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 151000, avg score 80.1, perfect games 0%.

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

152 evals so far. Full series in [`b10c-disc995seed3_evals.json`](b10c-disc995seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 1.9 | 1.9 | 0 | 5/95 | 0.89 | 0 | 0.4 |
| 2000 | 2.7 | 2.3 | 0 | 8/95 | 1.691 | 0 | 0.4 |
| ... | | | | | | | |
| 140000 | 81.8 | 80.72 | 73 | 89/95 | 80.286 | 0 | 0.001 |
| 141000 | 83.1 | 81.1 | 71 | 92/95 | 81.635 | 0 | 0.001 |
| 142000 | 77.9 | 80.76 | 70 | 92/95 | 76.515 | 0 | 0.001 |
| 143000 | 77.0 | 79.8 | 65 | 87/95 | 75.609 | 0 | 0.001 |
| 144000 | 78.0 | 79.56 | 71 | 86/95 | 76.615 | 0 | 0.001 |
| 145000 | 80.8 | 79.36 | 71 | 87/95 | 79.358 | 0 | 0.001 |
| 146000 | 81.0 | 78.94 | 70 | 87/95 | 79.423 | 0 | 0.001 |
| 147000 | 81.1 | 79.58 | 71 | 93/95 | 79.712 | 0 | 0.001 |
| 148000 | 80.3 | 80.24 | 71 | 88/95 | 78.941 | 0 | 0.001 |
| 149000 | 82.1 | 81.06 | 79 | 84/95 | 80.706 | 0 | 0.001 |
| 150000 | 80.4 | 80.98 | 70 | 88/95 | 79.031 | 0 | 0.001 |
| 151000 | 80.1 | 80.8 | 73 | 92/95 | 78.679 | 0 | 0.001 |
