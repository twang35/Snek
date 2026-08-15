# b30h-chase10fc200x100x100seed4

![b30h-chase10fc200x100x100seed4 progress](b30h-chase10fc200x100x100seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 93.0, perfect games 70%.

## Config

| setting | value |
|---|---|
| policy_name | b30h-chase10fc200x100x100seed4 |
| seed | 4 |
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
| fc_layer_params | (200, 100, 100) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | td_error |
| importance_sampling_beta | disabled |
| max_steps | 2000000 |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.0 |
| CHASE_SAFE_SHAPING | c=0.1, potential-based on head/food/tail in one region, gated to snake length >= 85 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2001 evals so far. Full series in [`b30h-chase10fc200x100x100seed4_evals.json`](b30h-chase10fc200x100x100seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.6 | 0.6 | 0 | 3/95 | 0.1 | 0 | 0.4 |
| 2000 | 0.9 | 0.75 | 0 | 3/95 | 0.4 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 94.3 | 94.4 | 92 | 95/95 | 162.976 | 70 | 0.0022 |
| 1990000 | 95.0 | 94.58 | 95 | 95/95 | 193.98 | 100 | 0.0021 |
| 1991000 | 84.8 | 92.64 | 13 | 95/95 | 134.032 | 50 | 0.0022 |
| 1992000 | 94.7 | 92.76 | 93 | 95/95 | 172.88 | 80 | 0.0022 |
| 1993000 | 86.0 | 90.96 | 5 | 95/95 | 175.033 | 90 | 0.0021 |
| 1994000 | 94.1 | 90.92 | 86 | 95/95 | 183.129 | 90 | 0.0022 |
| 1995000 | 92.4 | 90.4 | 76 | 95/95 | 161.532 | 70 | 0.0022 |
| 1996000 | 93.9 | 92.22 | 90 | 95/95 | 152.628 | 60 | 0.0022 |
| 1997000 | 95.0 | 92.28 | 95 | 95/95 | 193.98 | 100 | 0.0022 |
| 1998000 | 95.0 | 94.08 | 95 | 95/95 | 193.978 | 100 | 0.0021 |
| 1999000 | 94.7 | 94.2 | 92 | 95/95 | 183.73 | 90 | 0.0021 |
| 2000000 | 93.0 | 94.32 | 79 | 95/95 | 161.684 | 70 | 0.0021 |
