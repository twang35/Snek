# b2a-base2

![b2a-base2 progress](b2a-base2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 999000, avg score 63.6, perfect games 20%.

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

1000 evals so far. Full series in [`b2a-base2_evals.json`](b2a-base2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 1/95 | -0.153 | 0 | 0.4 |
| 2000 | 1.1 | 0.75 | 0 | 4/95 | 0.543 | 0 | 0.4 |
| ... | | | | | | | |
| 988000 | 58.0 | 60.58 | 30 | 90/95 | 52.544 | 0 | 0.001 |
| 989000 | 63.1 | 61.7 | 41 | 93/95 | 57.625 | 0 | 0.001 |
| 990000 | 66.6 | 61.8 | 49 | 82/95 | 61.084 | 0 | 0.001 |
| 991000 | 65.7 | 63.66 | 13 | 95/95 | 70.546 | 10 | 0.001 |
| 992000 | 58.9 | 62.46 | 12 | 87/95 | 53.487 | 0 | 0.001 |
| 993000 | 56.5 | 62.16 | 26 | 93/95 | 51.138 | 0 | 0.001 |
| 994000 | 66.9 | 62.92 | 45 | 81/95 | 61.425 | 0 | 0.001 |
| 995000 | 64.0 | 62.4 | 19 | 88/95 | 58.578 | 0 | 0.001 |
| 996000 | 70.5 | 63.36 | 21 | 94/95 | 64.924 | 0 | 0.001 |
| 997000 | 66.5 | 64.88 | 39 | 84/95 | 60.908 | 0 | 0.001 |
| 998000 | 57.8 | 65.14 | 23 | 91/95 | 52.385 | 0 | 0.001 |
| 999000 | 63.6 | 64.48 | 38 | 95/95 | 78.97 | 20 | 0.001 |
