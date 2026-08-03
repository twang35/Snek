# b10d-disc995seed4

![b10d-disc995seed4 progress](b10d-disc995seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 4452000, avg score 88.5, perfect games 80%.

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

4453 evals so far. Full series in [`b10d-disc995seed4_evals.json`](b10d-disc995seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.004 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 2/95 | -0.152 | 0 | 0.4 |
| 2000 | 0.0 | 0.2 | 0 | 0/95 | -0.548 | 0 | 0.4 |
| ... | | | | | | | |
| 4441000 | 78.1 | 87.26 | 24 | 95/95 | 126.207 | 50 | 0.0 |
| 4442000 | 91.4 | 88.7 | 67 | 95/95 | 149.02 | 60 | 0.0 |
| 4443000 | 87.0 | 87.2 | 22 | 95/95 | 155.033 | 70 | 0.0 |
| 4444000 | 82.9 | 84.86 | 34 | 95/95 | 150.501 | 70 | 0.0 |
| 4445000 | 88.7 | 85.62 | 40 | 95/95 | 156.198 | 70 | 0.0 |
| 4446000 | 93.2 | 88.64 | 85 | 95/95 | 150.77 | 60 | 0.0 |
| 4447000 | 79.3 | 86.22 | 33 | 95/95 | 147.033 | 70 | 0.0 |
| 4448000 | 84.0 | 85.62 | 21 | 95/95 | 110.754 | 30 | 0.0 |
| 4449000 | 88.1 | 86.66 | 29 | 95/95 | 166.191 | 80 | 0.0 |
| 4450000 | 81.5 | 85.22 | 34 | 95/95 | 118.753 | 40 | 0.0 |
| 4451000 | 91.1 | 84.8 | 59 | 95/95 | 159.082 | 70 | 0.0 |
| 4452000 | 88.5 | 86.64 | 36 | 95/95 | 166.881 | 80 | 0.0 |
