# b10a-disc995seed1

![b10a-disc995seed1 progress](b10a-disc995seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 39000, avg score 14.6, perfect games 0%.

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

40 evals so far. Full series in [`b10a-disc995seed1_evals.json`](b10a-disc995seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.901 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | -2.531 | 0 | 0.4 |
| 2000 | 0.7 | 0.7 | 0 | 2/95 | 0.142 | 0 | 0.4 |
| ... | | | | | | | |
| 28000 | 1.8 | 21.64 | 0 | 5/95 | 1.239 | 0 | 0.001 |
| 29000 | 3.6 | 11.14 | 0 | 15/95 | 3.017 | 0 | 0.001 |
| 30000 | 1.6 | 4.48 | 0 | 6/95 | 1.043 | 0 | 0.001 |
| 31000 | 1.1 | 3.42 | 0 | 4/95 | 0.546 | 0 | 0.001 |
| 32000 | 1.5 | 1.92 | 0 | 5/95 | 0.941 | 0 | 0.001 |
| 33000 | 1.9 | 1.94 | 0 | 5/95 | 1.346 | 0 | 0.001 |
| 34000 | 2.7 | 1.76 | 0 | 8/95 | 2.131 | 0 | 0.001 |
| 35000 | 6.1 | 2.66 | 2 | 8/95 | 5.505 | 0 | 0.001 |
| 36000 | 8.5 | 4.14 | 1 | 41/95 | 7.808 | 0 | 0.001 |
| 37000 | 3.8 | 4.6 | 1 | 21/95 | 3.214 | 0 | 0.001 |
| 38000 | 8.9 | 6.0 | 0 | 29/95 | 8.232 | 0 | 0.001 |
| 39000 | 14.6 | 8.38 | 2 | 36/95 | 13.838 | 0 | 0.001 |
