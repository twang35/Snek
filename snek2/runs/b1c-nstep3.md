# b1c-nstep3

![b1c-nstep3 progress](b1c-nstep3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 160000, avg score 42.3, perfect games 0%.

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

161 evals so far. Full series in [`b1c-nstep3_evals.json`](b1c-nstep3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.2 | 0.2 | 0 | 1/95 | -4.808 | 0 | 0.4 |
| 1000 | 1.8 | 1.8 | 0 | 5/95 | -1.0 | 0 | 0.4 |
| 2000 | 0.0 | 0.9 | 0 | 0/95 | -5.005 | 0 | 0.4 |
| ... | | | | | | | |
| 149000 | 46.7 | 41.18 | 28 | 88/95 | 41.459 | 0 | 0.01 |
| 150000 | 35.8 | 40.42 | 14 | 53/95 | 30.65 | 0 | 0.01 |
| 151000 | 42.6 | 43.2 | 22 | 76/95 | 37.372 | 0 | 0.01 |
| 152000 | 44.2 | 43.38 | 8 | 66/95 | 39.0 | 0 | 0.01 |
| 153000 | 47.9 | 43.44 | 23 | 74/95 | 42.556 | 0 | 0.01 |
| 154000 | 40.6 | 42.22 | 21 | 67/95 | 35.402 | 0 | 0.01 |
| 155000 | 42.9 | 43.64 | 12 | 87/95 | 38.074 | 0 | 0.01 |
| 156000 | 51.8 | 45.48 | 33 | 74/95 | 46.974 | 0 | 0.01 |
| 157000 | 54.2 | 47.48 | 32 | 82/95 | 48.841 | 0 | 0.01 |
| 158000 | 66.3 | 51.16 | 34 | 90/95 | 60.798 | 0 | 0.001 |
| 159000 | 45.1 | 52.06 | 22 | 86/95 | 39.859 | 0 | 0.001 |
| 160000 | 42.3 | 51.94 | 25 | 78/95 | 37.102 | 0 | 0.001 |
