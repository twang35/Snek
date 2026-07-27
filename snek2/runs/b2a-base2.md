# b2a-base2

![b2a-base2 progress](b2a-base2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 474000, avg score 66.2, perfect games 0%.

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

475 evals so far. Full series in [`b2a-base2_evals.json`](b2a-base2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 1/95 | -0.153 | 0 | 0.4 |
| 2000 | 1.1 | 0.75 | 0 | 4/95 | 0.543 | 0 | 0.4 |
| ... | | | | | | | |
| 463000 | 65.7 | 68.18 | 48 | 94/95 | 60.201 | 0 | 0.001 |
| 464000 | 54.6 | 64.84 | 13 | 86/95 | 49.205 | 0 | 0.001 |
| 465000 | 59.7 | 62.74 | 37 | 95/95 | 64.721 | 10 | 0.001 |
| 466000 | 68.1 | 61.48 | 38 | 88/95 | 62.583 | 0 | 0.001 |
| 467000 | 63.4 | 62.3 | 19 | 91/95 | 57.951 | 0 | 0.001 |
| 468000 | 64.2 | 62.0 | 31 | 94/95 | 58.668 | 0 | 0.001 |
| 469000 | 75.8 | 66.24 | 49 | 95/95 | 80.665 | 10 | 0.001 |
| 470000 | 76.6 | 69.62 | 52 | 92/95 | 70.748 | 0 | 0.001 |
| 471000 | 72.4 | 70.48 | 52 | 95/95 | 77.259 | 10 | 0.001 |
| 472000 | 60.5 | 69.9 | 19 | 75/95 | 55.131 | 0 | 0.001 |
| 473000 | 72.5 | 71.56 | 50 | 89/95 | 66.906 | 0 | 0.001 |
| 474000 | 66.2 | 69.64 | 38 | 90/95 | 60.735 | 0 | 0.001 |
