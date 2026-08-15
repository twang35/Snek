# b28d-chase20g85seed4

![b28d-chase20g85seed4 progress](b28d-chase20g85seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 94.1, perfect games 80%.

## Config

| setting | value |
|---|---|
| policy_name | b28d-chase20g85seed4 |
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
| fc_layer_params | (320,) |
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
| CHASE_SAFE_SHAPING | c=0.2, potential-based on head/food/tail in one region, gated to snake length >= 85 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2001 evals so far. Full series in [`b28d-chase20g85seed4_evals.json`](b28d-chase20g85seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 2.3 | 2.3 | 0 | 13/95 | 1.8 | 0 | 0.4 |
| 2000 | 6.5 | 4.4 | 1 | 29/95 | 6.0 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 94.3 | 94.7 | 88 | 95/95 | 183.312 | 90 | 0.002 |
| 1990000 | 94.3 | 94.56 | 88 | 95/95 | 183.312 | 90 | 0.002 |
| 1991000 | 93.1 | 94.18 | 76 | 95/95 | 182.116 | 90 | 0.002 |
| 1992000 | 95.0 | 94.18 | 95 | 95/95 | 193.966 | 100 | 0.002 |
| 1993000 | 95.0 | 94.34 | 95 | 95/95 | 193.967 | 100 | 0.002 |
| 1994000 | 94.7 | 94.42 | 92 | 95/95 | 183.713 | 90 | 0.002 |
| 1995000 | 91.9 | 93.94 | 64 | 95/95 | 180.921 | 90 | 0.002 |
| 1996000 | 91.7 | 93.66 | 74 | 95/95 | 160.814 | 70 | 0.002 |
| 1997000 | 89.1 | 92.48 | 36 | 95/95 | 177.67 | 90 | 0.002 |
| 1998000 | 93.6 | 92.2 | 81 | 95/95 | 182.607 | 90 | 0.002 |
| 1999000 | 95.0 | 92.26 | 95 | 95/95 | 193.961 | 100 | 0.002 |
| 2000000 | 94.1 | 92.7 | 88 | 95/95 | 173.167 | 80 | 0.002 |
