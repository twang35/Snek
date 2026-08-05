# b12a-eps002seed1

![b12a-eps002seed1 progress](b12a-eps002seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1121000, avg score 61.2, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b12a-eps002seed1 |
| seed | 1 |
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

1122 evals so far. Full series in [`b12a-eps002seed1_evals.json`](b12a-eps002seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 1000 | 0.3 | 0.3 | 0 | 3/95 | -0.697 | 0 | 0.4 |
| 2000 | 0.0 | 0.15 | 0 | 0/95 | -0.549 | 0 | 0.4 |
| ... | | | | | | | |
| 1110000 | 54.7 | 56.12 | 29 | 64/95 | 51.83 | 0 | 0.05 |
| 1111000 | 53.6 | 56.34 | 17 | 66/95 | 51.175 | 0 | 0.05 |
| 1112000 | 54.8 | 55.84 | 7 | 64/95 | 51.99 | 0 | 0.05 |
| 1113000 | 57.1 | 55.96 | 34 | 68/95 | 54.589 | 0 | 0.05 |
| 1114000 | 60.1 | 56.06 | 53 | 66/95 | 57.662 | 0 | 0.05 |
| 1115000 | 61.4 | 57.4 | 54 | 68/95 | 57.106 | 0 | 0.05 |
| 1116000 | 60.8 | 58.84 | 53 | 68/95 | 57.864 | 0 | 0.05 |
| 1117000 | 59.3 | 59.74 | 50 | 67/95 | 56.844 | 0 | 0.05 |
| 1118000 | 63.1 | 60.94 | 57 | 68/95 | 59.708 | 0 | 0.05 |
| 1119000 | 54.7 | 59.86 | 5 | 66/95 | 51.721 | 0 | 0.05 |
| 1120000 | 59.9 | 59.56 | 53 | 68/95 | 57.037 | 0 | 0.05 |
| 1121000 | 61.2 | 59.64 | 44 | 68/95 | 56.199 | 0 | 0.05 |
