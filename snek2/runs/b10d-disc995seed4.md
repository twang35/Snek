# b10d-disc995seed4

![b10d-disc995seed4 progress](b10d-disc995seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 146000, avg score 87.5, perfect games 0%.

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

147 evals so far. Full series in [`b10d-disc995seed4_evals.json`](b10d-disc995seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.004 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 2/95 | -0.152 | 0 | 0.4 |
| 2000 | 0.0 | 0.2 | 0 | 0/95 | -0.548 | 0 | 0.4 |
| ... | | | | | | | |
| 135000 | 86.2 | 85.78 | 81 | 91/95 | 84.806 | 0 | 0.001 |
| 136000 | 86.1 | 86.28 | 82 | 92/95 | 84.732 | 0 | 0.001 |
| 137000 | 86.5 | 86.34 | 83 | 90/95 | 85.092 | 0 | 0.001 |
| 138000 | 86.0 | 86.24 | 75 | 89/95 | 84.525 | 0 | 0.001 |
| 139000 | 85.6 | 86.08 | 81 | 89/95 | 84.118 | 0 | 0.001 |
| 140000 | 89.9 | 86.82 | 85 | 95/95 | 98.312 | 10 | 0.001 |
| 141000 | 77.7 | 85.14 | 9 | 93/95 | 76.391 | 0 | 0.001 |
| 142000 | 87.4 | 85.32 | 78 | 95/95 | 95.867 | 10 | 0.001 |
| 143000 | 87.4 | 85.6 | 80 | 95/95 | 95.829 | 10 | 0.001 |
| 144000 | 86.1 | 85.7 | 82 | 92/95 | 84.704 | 0 | 0.001 |
| 145000 | 85.4 | 84.8 | 79 | 92/95 | 83.886 | 0 | 0.001 |
| 146000 | 87.5 | 86.76 | 83 | 90/95 | 85.983 | 0 | 0.001 |
