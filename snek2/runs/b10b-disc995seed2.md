# b10b-disc995seed2

![b10b-disc995seed2 progress](b10b-disc995seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 27000, avg score 71.0, perfect games 10%.

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

28 evals so far. Full series in [`b10b-disc995seed2_evals.json`](b10b-disc995seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.7 | 0.7 | 0 | 3/95 | -4.301 | 0 | 0.4 |
| 1000 | 1.1 | 1.1 | 0 | 4/95 | 0.083 | 0 | 0.4 |
| 2000 | 7.6 | 4.35 | 0 | 24/95 | 6.516 | 0 | 0.2 |
| ... | | | | | | | |
| 16000 | 81.5 | 71.86 | 59 | 92/95 | 80.028 | 0 | 0.001 |
| 17000 | 73.2 | 71.74 | 2 | 95/95 | 81.669 | 10 | 0.001 |
| 18000 | 69.1 | 74.8 | 2 | 91/95 | 67.094 | 0 | 0.001 |
| 19000 | 41.5 | 68.2 | 2 | 93/95 | 40.375 | 0 | 0.001 |
| 20000 | 48.1 | 62.68 | 2 | 92/95 | 46.452 | 0 | 0.001 |
| 21000 | 52.1 | 56.8 | 2 | 93/95 | 50.812 | 0 | 0.001 |
| 22000 | 57.4 | 53.64 | 2 | 95/95 | 65.932 | 10 | 0.001 |
| 23000 | 54.4 | 50.7 | 2 | 95/95 | 73.126 | 20 | 0.001 |
| 24000 | 35.6 | 49.52 | 2 | 92/95 | 34.564 | 0 | 0.001 |
| 25000 | 30.8 | 46.06 | 2 | 82/95 | 29.846 | 0 | 0.001 |
| 26000 | 79.5 | 51.54 | 53 | 95/95 | 87.418 | 10 | 0.001 |
| 27000 | 71.0 | 54.26 | 7 | 95/95 | 79.427 | 10 | 0.001 |
