# b10b-disc995seed2

![b10b-disc995seed2 progress](b10b-disc995seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 158000, avg score 79.1, perfect games 30%.

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

159 evals so far. Full series in [`b10b-disc995seed2_evals.json`](b10b-disc995seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.7 | 0.7 | 0 | 3/95 | -4.301 | 0 | 0.4 |
| 1000 | 1.1 | 1.1 | 0 | 4/95 | 0.083 | 0 | 0.4 |
| 2000 | 7.6 | 4.35 | 0 | 24/95 | 6.516 | 0 | 0.2 |
| ... | | | | | | | |
| 147000 | 84.8 | 74.94 | 33 | 95/95 | 142.094 | 60 | 0.0 |
| 148000 | 72.7 | 73.44 | 21 | 95/95 | 89.727 | 20 | 0.0 |
| 149000 | 81.7 | 79.52 | 32 | 95/95 | 109.116 | 30 | 0.0 |
| 150000 | 91.1 | 83.1 | 58 | 95/95 | 169.166 | 80 | 0.0 |
| 151000 | 87.2 | 83.5 | 55 | 95/95 | 125.396 | 40 | 0.0 |
| 152000 | 85.1 | 83.56 | 60 | 95/95 | 122.806 | 40 | 0.0 |
| 153000 | 82.8 | 85.58 | 40 | 95/95 | 110.15 | 30 | 0.0 |
| 154000 | 94.0 | 88.04 | 90 | 95/95 | 162.457 | 70 | 0.0 |
| 155000 | 84.2 | 86.66 | 9 | 95/95 | 132.841 | 50 | 0.0 |
| 156000 | 91.0 | 87.42 | 79 | 95/95 | 129.629 | 40 | 0.0 |
| 157000 | 91.7 | 88.74 | 76 | 95/95 | 159.807 | 70 | 0.0 |
| 158000 | 79.1 | 88.0 | 22 | 95/95 | 106.996 | 30 | 0.0 |
