# b2a-base2

![b2a-base2 progress](b2a-base2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 768000, avg score 63.7, perfect games 10%.

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

769 evals so far. Full series in [`b2a-base2_evals.json`](b2a-base2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 1/95 | -0.153 | 0 | 0.4 |
| 2000 | 1.1 | 0.75 | 0 | 4/95 | 0.543 | 0 | 0.4 |
| ... | | | | | | | |
| 757000 | 63.1 | 65.16 | 34 | 90/95 | 57.656 | 0 | 0.001 |
| 758000 | 69.4 | 65.1 | 37 | 89/95 | 63.884 | 0 | 0.001 |
| 759000 | 66.2 | 66.58 | 24 | 86/95 | 60.733 | 0 | 0.001 |
| 760000 | 71.6 | 66.44 | 50 | 85/95 | 66.002 | 0 | 0.001 |
| 761000 | 63.3 | 66.72 | 44 | 90/95 | 57.792 | 0 | 0.001 |
| 762000 | 66.7 | 67.44 | 43 | 95/95 | 71.616 | 10 | 0.001 |
| 763000 | 66.5 | 66.86 | 15 | 84/95 | 61.059 | 0 | 0.001 |
| 764000 | 67.5 | 67.12 | 13 | 91/95 | 62.014 | 0 | 0.001 |
| 765000 | 73.2 | 67.44 | 49 | 95/95 | 78.102 | 10 | 0.001 |
| 766000 | 62.9 | 67.36 | 38 | 93/95 | 57.45 | 0 | 0.001 |
| 767000 | 71.0 | 68.22 | 38 | 84/95 | 65.537 | 0 | 0.001 |
| 768000 | 63.7 | 67.66 | 20 | 95/95 | 68.595 | 10 | 0.001 |
