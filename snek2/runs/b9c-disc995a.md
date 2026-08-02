# b9c-disc995a

![b9c-disc995a progress](b9c-disc995a.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3519000, avg score 73.5, perfect games 0%.

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

3520 evals so far. Full series in [`b9c-disc995a_evals.json`](b9c-disc995a_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -5.023 | 0 | 0.4 |
| 2000 | 0.2 | 0.1 | 0 | 1/95 | -3.022 | 0 | 0.4 |
| ... | | | | | | | |
| 3508000 | 72.1 | 74.2 | 52 | 91/95 | 66.572 | 0 | 0.0 |
| 3509000 | 89.7 | 77.88 | 78 | 95/95 | 125.617 | 40 | 0.0 |
| 3510000 | 77.4 | 76.64 | 52 | 92/95 | 71.84 | 0 | 0.0 |
| 3511000 | 80.0 | 80.72 | 64 | 95/95 | 84.76 | 10 | 0.0 |
| 3512000 | 66.7 | 77.18 | 5 | 95/95 | 92.516 | 30 | 0.0 |
| 3513000 | 72.5 | 77.26 | 21 | 95/95 | 87.885 | 20 | 0.0 |
| 3514000 | 82.8 | 75.88 | 51 | 95/95 | 118.841 | 40 | 0.0 |
| 3515000 | 69.8 | 74.36 | 35 | 90/95 | 64.324 | 0 | 0.0 |
| 3516000 | 77.8 | 73.92 | 39 | 95/95 | 82.599 | 10 | 0.0 |
| 3517000 | 81.1 | 76.8 | 56 | 95/95 | 117.206 | 40 | 0.0 |
| 3518000 | 77.6 | 77.82 | 48 | 95/95 | 103.305 | 30 | 0.0 |
| 3519000 | 73.5 | 75.96 | 5 | 91/95 | 67.936 | 0 | 0.0 |
