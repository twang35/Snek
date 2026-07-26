# b1c-nstep3

![b1c-nstep3 progress](b1c-nstep3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 217000, avg score 54.8, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b1c-nstep3 |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.99 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 3 |
| initial_epsilon | 0.4 |
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| importance_sampling_beta | 0.4 -> 1.0 over 1000000 steps |
| initial_populate_steps | 1000 |
| initialize_with_schmid | False |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |

## Evals

218 evals so far. Full series in [`b1c-nstep3_evals.json`](b1c-nstep3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.2 | 0.2 | 0 | 1/95 | -4.808 | 0 | 0.4 |
| 1000 | 1.8 | 1.8 | 0 | 5/95 | -1.0 | 0 | 0.4 |
| 2000 | 0.0 | 0.9 | 0 | 0/95 | -5.005 | 0 | 0.4 |
| ... | | | | | | | |
| 206000 | 60.2 | 61.18 | 33 | 95/95 | 65.148 | 10 | 0.001 |
| 207000 | 57.9 | 60.5 | 15 | 84/95 | 52.496 | 0 | 0.001 |
| 208000 | 60.6 | 60.6 | 29 | 86/95 | 55.197 | 0 | 0.001 |
| 209000 | 49.0 | 57.82 | 18 | 72/95 | 43.728 | 0 | 0.001 |
| 210000 | 60.2 | 57.58 | 28 | 80/95 | 54.837 | 0 | 0.001 |
| 211000 | 73.6 | 60.26 | 49 | 92/95 | 68.407 | 0 | 0.001 |
| 212000 | 55.0 | 59.68 | 6 | 78/95 | 49.532 | 0 | 0.001 |
| 213000 | 67.7 | 61.1 | 44 | 83/95 | 62.144 | 0 | 0.001 |
| 214000 | 71.7 | 65.64 | 36 | 93/95 | 66.917 | 0 | 0.001 |
| 215000 | 65.4 | 66.68 | 34 | 82/95 | 59.894 | 0 | 0.001 |
| 216000 | 71.7 | 66.3 | 48 | 93/95 | 66.201 | 0 | 0.001 |
| 217000 | 54.8 | 66.26 | 26 | 78/95 | 49.392 | 0 | 0.001 |
