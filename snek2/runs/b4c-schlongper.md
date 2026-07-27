# b4c-schlongper

![b4c-schlongper progress](b4c-schlongper.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1096000, avg score 87.0, perfect games 30%.

## Config

| setting | value |
|---|---|
| policy_name | b4c-schlongper |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.99 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
| initial_epsilon | 0.4 |
| min_epsilon | 0.0 |
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.8 |
| priority_signal | td_loss |
| importance_sampling_beta | disabled |
| initial_populate_steps | 1000 |
| initialize_with_schmid | False |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |

## Evals

1097 evals so far. Full series in [`b4c-schlongper_evals.json`](b4c-schlongper_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 4/95 | -4.316 | 0 | 0.4 |
| 2000 | 0.5 | 0.6 | 0 | 4/95 | -4.55 | 0 | 0.4 |
| ... | | | | | | | |
| 1085000 | 72.6 | 70.26 | 54 | 95/95 | 77.422 | 10 | 0.0 |
| 1086000 | 85.8 | 74.72 | 73 | 95/95 | 111.19 | 30 | 0.0 |
| 1087000 | 83.5 | 75.7 | 59 | 95/95 | 98.374 | 20 | 0.0 |
| 1088000 | 72.3 | 77.56 | 48 | 94/95 | 66.657 | 0 | 0.0 |
| 1089000 | 73.8 | 77.6 | 41 | 95/95 | 78.493 | 10 | 0.0 |
| 1090000 | 67.8 | 76.64 | 38 | 95/95 | 72.547 | 10 | 0.0 |
| 1091000 | 75.4 | 74.56 | 24 | 95/95 | 100.94 | 30 | 0.0 |
| 1092000 | 78.2 | 73.5 | 23 | 95/95 | 103.472 | 30 | 0.0 |
| 1093000 | 79.9 | 75.02 | 44 | 95/95 | 105.471 | 30 | 0.0 |
| 1094000 | 74.3 | 75.12 | 34 | 95/95 | 89.452 | 20 | 0.0 |
| 1095000 | 73.5 | 76.26 | 22 | 95/95 | 99.076 | 30 | 0.0 |
| 1096000 | 87.0 | 78.58 | 45 | 95/95 | 112.322 | 30 | 0.0 |
