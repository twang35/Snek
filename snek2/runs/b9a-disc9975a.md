# b9a-disc9975a

![b9a-disc9975a progress](b9a-disc9975a.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3552000, avg score 71.6, perfect games 20%.

## Config

| setting | value |
|---|---|
| policy_name | b9a-disc9975a |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.9975 |
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

3553 evals so far. Full series in [`b9a-disc9975a_evals.json`](b9a-disc9975a_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.1 | 0.1 | 0 | 1/95 | -4.903 | 0 | 0.4 |
| 2000 | 0.5 | 0.3 | 0 | 2/95 | -4.55 | 0 | 0.4 |
| ... | | | | | | | |
| 3541000 | 33.6 | 54.76 | 4 | 95/95 | 49.163 | 20 | 0.0 |
| 3542000 | 76.6 | 56.04 | 19 | 95/95 | 91.779 | 20 | 0.0 |
| 3543000 | 81.4 | 69.52 | 19 | 95/95 | 106.86 | 30 | 0.0 |
| 3544000 | 69.0 | 66.5 | 11 | 95/95 | 94.625 | 30 | 0.0 |
| 3545000 | 59.6 | 64.04 | 4 | 95/95 | 95.689 | 40 | 0.0 |
| 3546000 | 85.5 | 74.42 | 19 | 95/95 | 142.352 | 60 | 0.0 |
| 3547000 | 87.0 | 76.5 | 60 | 95/95 | 133.51 | 50 | 0.0 |
| 3548000 | 79.6 | 76.14 | 13 | 95/95 | 126.168 | 50 | 0.0 |
| 3549000 | 78.7 | 78.08 | 11 | 95/95 | 83.538 | 10 | 0.0 |
| 3550000 | 90.2 | 84.2 | 74 | 95/95 | 126.207 | 40 | 0.0 |
| 3551000 | 83.9 | 83.88 | 56 | 95/95 | 130.352 | 50 | 0.0 |
| 3552000 | 71.6 | 80.8 | 13 | 95/95 | 86.797 | 20 | 0.0 |
