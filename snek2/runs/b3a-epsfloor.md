# b3a-epsfloor

![b3a-epsfloor progress](b3a-epsfloor.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 255000, avg score 70.4, perfect games 10%.

## Config

| setting | value |
|---|---|
| policy_name | b3a-epsfloor |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.99 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
| initial_epsilon | 0.4 |
| min_epsilon | 0.001 |
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

256 evals so far. Full series in [`b3a-epsfloor_evals.json`](b3a-epsfloor_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.2 | 0.2 | 0 | 1/95 | -4.849 | 0 | 0.4 |
| 2000 | 0.3 | 0.25 | 0 | 2/95 | -4.754 | 0 | 0.4 |
| ... | | | | | | | |
| 244000 | 74.2 | 73.06 | 52 | 95/95 | 78.94 | 10 | 0.001 |
| 245000 | 67.4 | 71.8 | 49 | 90/95 | 61.906 | 0 | 0.001 |
| 246000 | 66.6 | 70.06 | 17 | 93/95 | 61.942 | 0 | 0.001 |
| 247000 | 82.9 | 72.1 | 58 | 95/95 | 88.482 | 10 | 0.001 |
| 248000 | 66.6 | 71.54 | 29 | 85/95 | 61.891 | 0 | 0.001 |
| 249000 | 74.4 | 71.58 | 42 | 95/95 | 90.025 | 20 | 0.001 |
| 250000 | 72.5 | 72.6 | 12 | 95/95 | 77.706 | 10 | 0.001 |
| 251000 | 82.1 | 75.7 | 56 | 95/95 | 86.899 | 10 | 0.001 |
| 252000 | 76.0 | 74.32 | 44 | 95/95 | 91.036 | 20 | 0.001 |
| 253000 | 75.5 | 76.1 | 58 | 91/95 | 70.299 | 0 | 0.001 |
| 254000 | 74.1 | 76.04 | 48 | 84/95 | 68.465 | 0 | 0.001 |
| 255000 | 70.4 | 75.62 | 37 | 95/95 | 75.248 | 10 | 0.001 |
