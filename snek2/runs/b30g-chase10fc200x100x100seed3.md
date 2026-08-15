# b30g-chase10fc200x100x100seed3

![b30g-chase10fc200x100x100seed3 progress](b30g-chase10fc200x100x100seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 94.6, perfect games 80%.

## Config

| setting | value |
|---|---|
| policy_name | b30g-chase10fc200x100x100seed3 |
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

2001 evals so far. Full series in [`b30g-chase10fc200x100x100seed3_evals.json`](b30g-chase10fc200x100x100seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 1.3 | 1.3 | 0 | 4/95 | 0.8 | 0 | 0.4 |
| 2000 | 1.0 | 1.15 | 0 | 5/95 | 0.5 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 89.2 | 93.72 | 42 | 95/95 | 157.882 | 70 | 0.002 |
| 1990000 | 94.8 | 93.68 | 93 | 95/95 | 183.828 | 90 | 0.002 |
| 1991000 | 94.8 | 93.68 | 93 | 95/95 | 183.831 | 90 | 0.002 |
| 1992000 | 94.8 | 93.72 | 93 | 95/95 | 183.832 | 90 | 0.002 |
| 1993000 | 94.8 | 93.68 | 93 | 95/95 | 183.83 | 90 | 0.002 |
| 1994000 | 95.0 | 94.84 | 95 | 95/95 | 193.982 | 100 | 0.002 |
| 1995000 | 88.3 | 93.54 | 34 | 95/95 | 146.582 | 60 | 0.002 |
| 1996000 | 94.8 | 93.54 | 93 | 95/95 | 183.827 | 90 | 0.002 |
| 1997000 | 87.3 | 92.04 | 24 | 95/95 | 146.48 | 60 | 0.002 |
| 1998000 | 94.8 | 92.04 | 93 | 95/95 | 183.381 | 90 | 0.002 |
| 1999000 | 94.6 | 91.96 | 93 | 95/95 | 173.229 | 80 | 0.002 |
| 2000000 | 94.6 | 93.22 | 93 | 95/95 | 173.678 | 80 | 0.002 |
