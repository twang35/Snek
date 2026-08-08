# b16b-noshapeseed2

![b16b-noshapeseed2 progress](b16b-noshapeseed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1261000, avg score 93.5, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | b16b-noshapeseed2 |
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

1262 evals so far. Full series in [`b16b-noshapeseed2_evals.json`](b16b-noshapeseed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.8 | 0.8 | 0 | 3/95 | -0.15 | 0 | 0.4 |
| 2000 | 0.7 | 0.75 | 0 | 2/95 | 0.2 | 0 | 0.4 |
| ... | | | | | | | |
| 1250000 | 92.6 | 94.0 | 88 | 95/95 | 151.8 | 60 | 0.002 |
| 1251000 | 93.9 | 93.78 | 90 | 95/95 | 162.6 | 70 | 0.002 |
| 1252000 | 94.4 | 93.94 | 92 | 95/95 | 163.1 | 70 | 0.002 |
| 1253000 | 94.4 | 93.88 | 92 | 95/95 | 173.5 | 80 | 0.0021 |
| 1254000 | 95.0 | 94.06 | 95 | 95/95 | 194.0 | 100 | 0.0021 |
| 1255000 | 93.2 | 94.18 | 85 | 95/95 | 141.55 | 50 | 0.0021 |
| 1256000 | 94.4 | 94.28 | 90 | 95/95 | 173.05 | 80 | 0.0021 |
| 1257000 | 95.0 | 94.4 | 95 | 95/95 | 194.0 | 100 | 0.0021 |
| 1258000 | 94.5 | 94.42 | 92 | 95/95 | 173.6 | 80 | 0.0021 |
| 1259000 | 95.0 | 94.42 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 1260000 | 94.7 | 94.72 | 92 | 95/95 | 183.75 | 90 | 0.002 |
| 1261000 | 93.5 | 94.54 | 88 | 95/95 | 152.7 | 60 | 0.002 |
