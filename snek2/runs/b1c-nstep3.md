# b1c-nstep3

![b1c-nstep3 progress](b1c-nstep3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1139000, avg score 26.6, perfect games 0%.

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

1140 evals so far. Full series in [`b1c-nstep3_evals.json`](b1c-nstep3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.2 | 0.2 | 0 | 1/95 | -4.808 | 0 | 0.4 |
| 1000 | 1.8 | 1.8 | 0 | 5/95 | -1.0 | 0 | 0.4 |
| 2000 | 0.0 | 0.9 | 0 | 0/95 | -5.005 | 0 | 0.4 |
| ... | | | | | | | |
| 1128000 | 29.2 | 27.18 | 20 | 34/95 | 24.145 | 0 | 0.001 |
| 1129000 | 24.4 | 26.78 | 9 | 34/95 | 19.35 | 0 | 0.001 |
| 1130000 | 27.8 | 27.02 | 22 | 34/95 | 22.729 | 0 | 0.001 |
| 1131000 | 27.4 | 27.02 | 16 | 38/95 | 22.338 | 0 | 0.001 |
| 1132000 | 29.1 | 27.58 | 11 | 34/95 | 24.039 | 0 | 0.001 |
| 1133000 | 25.8 | 26.9 | 14 | 34/95 | 20.741 | 0 | 0.001 |
| 1134000 | 25.0 | 27.02 | 16 | 32/95 | 19.941 | 0 | 0.001 |
| 1135000 | 26.6 | 26.78 | 19 | 31/95 | 21.55 | 0 | 0.001 |
| 1136000 | 24.9 | 26.28 | 11 | 31/95 | 19.834 | 0 | 0.001 |
| 1137000 | 26.9 | 25.84 | 12 | 32/95 | 21.851 | 0 | 0.001 |
| 1138000 | 27.8 | 26.24 | 21 | 31/95 | 22.73 | 0 | 0.001 |
| 1139000 | 26.6 | 26.56 | 12 | 38/95 | 21.547 | 0 | 0.001 |
