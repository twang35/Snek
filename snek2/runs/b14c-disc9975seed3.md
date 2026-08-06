# b14c-disc9975seed3

![b14c-disc9975seed3 progress](b14c-disc9975seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1341000, avg score 91.9, perfect games 40%.

## Config

| setting | value |
|---|---|
| policy_name | b14c-disc9975seed3 |
| seed | 3 |
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

1342 evals so far. Full series in [`b14c-disc9975seed3_evals.json`](b14c-disc9975seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.902 | 0 | 0.4 |
| 1000 | 1.6 | 1.6 | 0 | 4/95 | 1.031 | 0 | 0.4 |
| 2000 | 1.0 | 1.3 | 0 | 5/95 | 0.445 | 0 | 0.4 |
| ... | | | | | | | |
| 1330000 | 91.5 | 91.72 | 84 | 95/95 | 117.371 | 30 | 0.004 |
| 1331000 | 94.4 | 92.54 | 92 | 95/95 | 161.3 | 70 | 0.0039 |
| 1332000 | 89.7 | 92.66 | 67 | 95/95 | 94.635 | 10 | 0.0041 |
| 1333000 | 93.4 | 92.62 | 86 | 95/95 | 150.027 | 60 | 0.004 |
| 1334000 | 94.0 | 92.6 | 90 | 95/95 | 161.544 | 70 | 0.004 |
| 1335000 | 94.2 | 93.14 | 92 | 95/95 | 161.276 | 70 | 0.004 |
| 1336000 | 93.6 | 92.98 | 90 | 95/95 | 139.873 | 50 | 0.004 |
| 1337000 | 91.8 | 93.4 | 76 | 95/95 | 117.166 | 30 | 0.0041 |
| 1338000 | 90.8 | 92.88 | 78 | 95/95 | 126.578 | 40 | 0.0041 |
| 1339000 | 92.7 | 92.62 | 86 | 95/95 | 118.449 | 30 | 0.0041 |
| 1340000 | 93.1 | 92.4 | 86 | 95/95 | 160.329 | 70 | 0.004 |
| 1341000 | 91.9 | 92.06 | 80 | 95/95 | 127.636 | 40 | 0.004 |
