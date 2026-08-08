# b16c-noshapeseed3

![b16c-noshapeseed3 progress](b16c-noshapeseed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1257000, avg score 93.7, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | b16c-noshapeseed3 |
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
| max_steps | 10000000 |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.0 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

1258 evals so far. Full series in [`b16c-noshapeseed3_evals.json`](b16c-noshapeseed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 1.5 | 1.5 | 0 | 4/95 | 1.0 | 0 | 0.4 |
| 2000 | 0.9 | 1.2 | 0 | 5/95 | 0.4 | 0 | 0.4 |
| ... | | | | | | | |
| 1246000 | 58.0 | 86.48 | 0 | 95/95 | 117.2 | 60 | 0.0026 |
| 1247000 | 92.8 | 86.5 | 84 | 95/95 | 142.05 | 50 | 0.0027 |
| 1248000 | 92.5 | 86.0 | 80 | 95/95 | 151.7 | 60 | 0.0027 |
| 1249000 | 84.8 | 84.26 | 11 | 95/95 | 114.15 | 30 | 0.0028 |
| 1250000 | 94.7 | 84.56 | 92 | 95/95 | 183.75 | 90 | 0.0028 |
| 1251000 | 84.7 | 89.9 | 1 | 95/95 | 143.9 | 60 | 0.0029 |
| 1252000 | 94.0 | 90.14 | 92 | 95/95 | 153.2 | 60 | 0.0029 |
| 1253000 | 94.7 | 90.58 | 92 | 95/95 | 183.75 | 90 | 0.0028 |
| 1254000 | 92.8 | 92.18 | 78 | 95/95 | 171.45 | 80 | 0.0028 |
| 1255000 | 93.9 | 92.02 | 89 | 95/95 | 163.05 | 70 | 0.0027 |
| 1256000 | 93.3 | 93.74 | 86 | 95/95 | 152.5 | 60 | 0.0027 |
| 1257000 | 93.7 | 93.68 | 90 | 95/95 | 152.9 | 60 | 0.0027 |
