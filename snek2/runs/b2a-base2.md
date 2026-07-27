# b2a-base2

![b2a-base2 progress](b2a-base2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 464000, avg score 54.6, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b2a-base2 |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.99 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
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

465 evals so far. Full series in [`b2a-base2_evals.json`](b2a-base2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 1/95 | -0.153 | 0 | 0.4 |
| 2000 | 1.1 | 0.75 | 0 | 4/95 | 0.543 | 0 | 0.4 |
| ... | | | | | | | |
| 453000 | 70.7 | 67.94 | 41 | 88/95 | 65.144 | 0 | 0.001 |
| 454000 | 71.5 | 67.9 | 38 | 84/95 | 65.932 | 0 | 0.001 |
| 455000 | 71.0 | 67.66 | 58 | 84/95 | 65.442 | 0 | 0.001 |
| 456000 | 70.8 | 70.12 | 48 | 93/95 | 65.182 | 0 | 0.001 |
| 457000 | 66.5 | 70.1 | 51 | 89/95 | 61.017 | 0 | 0.001 |
| 458000 | 80.1 | 71.98 | 56 | 95/95 | 84.811 | 10 | 0.001 |
| 459000 | 71.3 | 71.94 | 33 | 95/95 | 76.119 | 10 | 0.001 |
| 460000 | 70.2 | 71.78 | 46 | 86/95 | 64.624 | 0 | 0.001 |
| 461000 | 74.4 | 72.5 | 29 | 93/95 | 68.768 | 0 | 0.001 |
| 462000 | 59.3 | 71.06 | 17 | 87/95 | 53.905 | 0 | 0.001 |
| 463000 | 65.7 | 68.18 | 48 | 94/95 | 60.201 | 0 | 0.001 |
| 464000 | 54.6 | 64.84 | 13 | 86/95 | 49.205 | 0 | 0.001 |
