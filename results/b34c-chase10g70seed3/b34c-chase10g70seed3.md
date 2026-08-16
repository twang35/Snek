# b34c-chase10g70seed3

![b34c-chase10g70seed3 progress](b34c-chase10g70seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 94.8, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b34c-chase10g70seed3 |
| seed | 3 |
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
| CHASE_SAFE_SHAPING | c=0.1, potential-based on head/food/tail in one region, gated to snake length >= 70 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2001 evals so far. Full series in [`b34c-chase10g70seed3_evals.json`](b34c-chase10g70seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 3.2 | 3.2 | 0 | 7/95 | 2.7 | 0 | 0.4 |
| 2000 | 3.7 | 3.45 | 0 | 14/95 | 3.2 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 94.7 | 94.56 | 92 | 95/95 | 183.683 | 90 | 0.002 |
| 1990000 | 94.6 | 94.54 | 93 | 95/95 | 173.64 | 80 | 0.002 |
| 1991000 | 95.0 | 94.54 | 95 | 95/95 | 193.931 | 100 | 0.002 |
| 1992000 | 94.7 | 94.48 | 92 | 95/95 | 183.691 | 90 | 0.002 |
| 1993000 | 94.3 | 94.66 | 88 | 95/95 | 183.283 | 90 | 0.002 |
| 1994000 | 94.8 | 94.68 | 93 | 95/95 | 183.787 | 90 | 0.002 |
| 1995000 | 95.0 | 94.76 | 95 | 95/95 | 193.943 | 100 | 0.002 |
| 1996000 | 94.8 | 94.72 | 93 | 95/95 | 183.8 | 90 | 0.002 |
| 1997000 | 94.8 | 94.74 | 93 | 95/95 | 183.789 | 90 | 0.002 |
| 1998000 | 95.0 | 94.88 | 95 | 95/95 | 193.935 | 100 | 0.002 |
| 1999000 | 95.0 | 94.92 | 95 | 95/95 | 193.943 | 100 | 0.002 |
| 2000000 | 94.8 | 94.88 | 93 | 95/95 | 183.781 | 90 | 0.002 |
