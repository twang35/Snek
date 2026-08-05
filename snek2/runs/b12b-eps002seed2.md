# b12b-eps002seed2

![b12b-eps002seed2 progress](b12b-eps002seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1031000, avg score 54.8, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b12b-eps002seed2 |
| seed | 2 |
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

1032 evals so far. Full series in [`b12b-eps002seed2_evals.json`](b12b-eps002seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.003 | 0 | 0.4 |
| 1000 | 0.8 | 0.8 | 0 | 3/95 | -0.201 | 0 | 0.4 |
| 2000 | 0.7 | 0.75 | 0 | 2/95 | 0.146 | 0 | 0.4 |
| ... | | | | | | | |
| 1020000 | 56.0 | 54.88 | 51 | 69/95 | 54.723 | 0 | 0.05 |
| 1021000 | 57.9 | 55.7 | 49 | 65/95 | 56.229 | 0 | 0.05 |
| 1022000 | 53.8 | 55.9 | 49 | 58/95 | 52.671 | 0 | 0.05 |
| 1023000 | 53.9 | 55.46 | 49 | 65/95 | 52.708 | 0 | 0.05 |
| 1024000 | 54.4 | 55.2 | 47 | 67/95 | 53.253 | 0 | 0.05 |
| 1025000 | 50.9 | 54.18 | 28 | 60/95 | 49.798 | 0 | 0.05 |
| 1026000 | 55.6 | 53.72 | 53 | 58/95 | 54.365 | 0 | 0.05 |
| 1027000 | 54.3 | 53.82 | 50 | 58/95 | 53.125 | 0 | 0.05 |
| 1028000 | 52.9 | 53.62 | 49 | 60/95 | 51.785 | 0 | 0.05 |
| 1029000 | 48.5 | 52.44 | 45 | 54/95 | 47.456 | 0 | 0.05 |
| 1030000 | 56.3 | 53.52 | 49 | 63/95 | 55.108 | 0 | 0.05 |
| 1031000 | 54.8 | 53.36 | 49 | 61/95 | 53.6 | 0 | 0.05 |
