# b14b-disc9975seed2

![b14b-disc9975seed2 progress](b14b-disc9975seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1378000, avg score 92.8, perfect games 40%.

## Config

| setting | value |
|---|---|
| policy_name | b14b-disc9975seed2 |
| seed | 2 |
| zeroed_observations | none |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.9975 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
| initial_epsilon | 0.4 |
| min_epsilon | 0.002 |
| epsilon_schedule | bootstrap on avg_reward [2, 5, 10, 15, 20] then geometric to floor by 80% trailing-30 perfect |
| guided_fraction | 0.8 |
| exploration_shield | 80% of refinement-phase episodes draw the epsilon move from non-fatal actions; greedy moves never shielded |
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | td_loss |
| importance_sampling_beta | disabled |
| max_steps | 5000000 |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

1379 evals so far. Full series in [`b14b-disc9975seed2_evals.json`](b14b-disc9975seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.003 | 0 | 0.4 |
| 1000 | 0.8 | 0.8 | 0 | 3/95 | -0.201 | 0 | 0.4 |
| 2000 | 0.7 | 0.75 | 0 | 2/95 | 0.146 | 0 | 0.4 |
| ... | | | | | | | |
| 1367000 | 93.3 | 91.6 | 86 | 95/95 | 150.009 | 60 | 0.0035 |
| 1368000 | 93.0 | 93.18 | 83 | 95/95 | 150.635 | 60 | 0.0034 |
| 1369000 | 93.4 | 93.34 | 86 | 95/95 | 160.47 | 70 | 0.0035 |
| 1370000 | 94.3 | 93.56 | 92 | 95/95 | 161.437 | 70 | 0.0035 |
| 1371000 | 84.5 | 91.7 | 1 | 95/95 | 152.152 | 70 | 0.0034 |
| 1372000 | 84.8 | 90.0 | 1 | 95/95 | 142.087 | 60 | 0.0034 |
| 1373000 | 91.3 | 89.66 | 77 | 95/95 | 128.354 | 40 | 0.0035 |
| 1374000 | 85.2 | 88.02 | 1 | 95/95 | 163.314 | 80 | 0.0034 |
| 1375000 | 92.0 | 87.56 | 82 | 95/95 | 128.57 | 40 | 0.0034 |
| 1376000 | 84.3 | 87.52 | 2 | 95/95 | 131.156 | 50 | 0.0034 |
| 1377000 | 94.0 | 89.36 | 92 | 95/95 | 150.619 | 60 | 0.0033 |
| 1378000 | 92.8 | 89.66 | 88 | 95/95 | 128.567 | 40 | 0.0034 |
