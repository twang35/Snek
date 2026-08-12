# b23b-beta01seed2

![b23b-beta01seed2 progress](b23b-beta01seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1027000, avg score 91.6, perfect games 70%.

## Config

| setting | value |
|---|---|
| policy_name | b23b-beta01seed2 |
| seed | 2 |
| zeroed_observations | none |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.9975 |
| target_update_period | 1000 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
| initial_epsilon | 0.4 |
| min_epsilon | 0.002 |
| epsilon_schedule | bootstrap on avg_reward [2, 5, 10, 15, 20] then geometric to floor by 80% trailing-30 perfect |
| guided_fraction | 0.8 |
| forking | up to 4 live branches including the main line, fork p=0.5 at length >= 85, branch capped at 60 steps, one branch advanced per iteration |
| exploration_shield | 80% of refinement-phase episodes draw the epsilon move from non-fatal actions; greedy moves never shielded |
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | td_error |
| importance_sampling_beta | 0.0 -> 0.1 over 300000 steps |
| max_steps | 3000000 |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.0 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

1028 evals so far. Full series in [`b23b-beta01seed2_evals.json`](b23b-beta01seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.9 | 0.9 | 0 | 3/95 | 0.4 | 0 | 0.4 |
| 2000 | 0.8 | 0.85 | 0 | 2/95 | 0.3 | 0 | 0.4 |
| ... | | | | | | | |
| 1016000 | 94.7 | 92.78 | 92 | 95/95 | 183.75 | 90 | 0.0021 |
| 1017000 | 94.1 | 92.94 | 86 | 95/95 | 183.15 | 90 | 0.0021 |
| 1018000 | 94.5 | 92.94 | 92 | 95/95 | 173.6 | 80 | 0.0021 |
| 1019000 | 95.0 | 94.58 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 1020000 | 94.2 | 94.5 | 92 | 95/95 | 163.35 | 70 | 0.0021 |
| 1021000 | 94.8 | 94.52 | 93 | 95/95 | 183.85 | 90 | 0.002 |
| 1022000 | 93.6 | 94.42 | 87 | 95/95 | 162.75 | 70 | 0.002 |
| 1023000 | 93.9 | 94.3 | 90 | 95/95 | 153.1 | 60 | 0.0021 |
| 1024000 | 94.2 | 94.14 | 92 | 95/95 | 163.35 | 70 | 0.0021 |
| 1025000 | 94.8 | 94.26 | 93 | 95/95 | 183.85 | 90 | 0.0021 |
| 1026000 | 94.2 | 94.14 | 90 | 95/95 | 173.3 | 80 | 0.0021 |
| 1027000 | 91.6 | 93.74 | 65 | 95/95 | 160.75 | 70 | 0.0021 |
