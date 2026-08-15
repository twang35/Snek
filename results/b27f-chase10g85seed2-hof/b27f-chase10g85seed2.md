# b27f-chase10g85seed2

![b27f-chase10g85seed2 progress](b27f-chase10g85seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 91.6, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | b27f-chase10g85seed2 |
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
| CHASE_SAFE_SHAPING | c=0.1, potential-based on head/food/tail in one region, gated to snake length >= 85 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2001 evals so far. Full series in [`b27f-chase10g85seed2_evals.json`](b27f-chase10g85seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -3.2 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 2/95 | 0.2 | 0 | 0.4 |
| 2000 | 1.2 | 0.95 | 0 | 3/95 | 0.7 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 95.0 | 87.06 | 95 | 95/95 | 193.979 | 100 | 0.002 |
| 1990000 | 93.6 | 90.62 | 85 | 95/95 | 162.73 | 70 | 0.002 |
| 1991000 | 94.8 | 92.64 | 93 | 95/95 | 183.83 | 90 | 0.002 |
| 1992000 | 93.8 | 92.5 | 90 | 95/95 | 162.481 | 70 | 0.002 |
| 1993000 | 94.5 | 94.34 | 92 | 95/95 | 173.581 | 80 | 0.002 |
| 1994000 | 85.0 | 92.34 | 0 | 95/95 | 164.084 | 80 | 0.002 |
| 1995000 | 94.8 | 92.58 | 93 | 95/95 | 183.833 | 90 | 0.002 |
| 1996000 | 93.9 | 92.4 | 90 | 95/95 | 163.029 | 70 | 0.002 |
| 1997000 | 95.0 | 92.64 | 95 | 95/95 | 193.98 | 100 | 0.002 |
| 1998000 | 92.9 | 92.32 | 76 | 95/95 | 171.982 | 80 | 0.002 |
| 1999000 | 84.0 | 92.12 | 1 | 95/95 | 143.182 | 60 | 0.002 |
| 2000000 | 91.6 | 91.48 | 78 | 95/95 | 150.781 | 60 | 0.002 |
