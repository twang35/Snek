# b12d-eps002seed4

![b12d-eps002seed4 progress](b12d-eps002seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1092000, avg score 62.8, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b12d-eps002seed4 |
| seed | 4 |
| zeroed_observations | none |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.995 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
| initial_epsilon | 0.4 |
| min_epsilon | 0.002 |
| epsilon_schedule | bootstrap on avg_reward [5, 10, 20] then geometric to floor by 80% trailing-30 perfect |
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

1093 evals so far. Full series in [`b12d-eps002seed4_evals.json`](b12d-eps002seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 0 | 1/95 | -4.503 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | 0.147 | 0 | 0.4 |
| 2000 | 0.7 | 0.7 | 0 | 2/95 | 0.146 | 0 | 0.4 |
| ... | | | | | | | |
| 1081000 | 59.2 | 61.96 | 54 | 72/95 | 57.887 | 0 | 0.05 |
| 1082000 | 61.6 | 61.38 | 53 | 71/95 | 60.312 | 0 | 0.05 |
| 1083000 | 63.4 | 61.68 | 47 | 74/95 | 62.031 | 0 | 0.05 |
| 1084000 | 62.7 | 62.3 | 52 | 72/95 | 60.491 | 0 | 0.05 |
| 1085000 | 65.6 | 62.5 | 56 | 73/95 | 63.736 | 0 | 0.05 |
| 1086000 | 65.4 | 63.74 | 34 | 78/95 | 63.901 | 0 | 0.05 |
| 1087000 | 63.3 | 64.08 | 53 | 71/95 | 61.965 | 0 | 0.05 |
| 1088000 | 61.7 | 63.74 | 40 | 72/95 | 60.257 | 0 | 0.05 |
| 1089000 | 63.1 | 63.82 | 43 | 70/95 | 60.899 | 0 | 0.05 |
| 1090000 | 67.6 | 64.22 | 53 | 80/95 | 65.629 | 0 | 0.05 |
| 1091000 | 61.2 | 63.38 | 52 | 76/95 | 59.799 | 0 | 0.05 |
| 1092000 | 62.8 | 63.28 | 55 | 72/95 | 61.446 | 0 | 0.05 |
