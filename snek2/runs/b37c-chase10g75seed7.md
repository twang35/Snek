# b37c-chase10g75seed7

![b37c-chase10g75seed7 progress](b37c-chase10g75seed7.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 77.6, perfect games 70%.

## Config

| setting | value |
|---|---|
| policy_name | b37c-chase10g75seed7 |
| seed | 7 |
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
| CHASE_SAFE_SHAPING | c=0.1, potential-based on head/food/tail in one region, gated to snake length >= 75 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2001 evals so far. Full series in [`b37c-chase10g75seed7_evals.json`](b37c-chase10g75seed7_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 1.2 | 1.2 | 0 | 4/95 | 0.7 | 0 | 0.4 |
| 2000 | 1.4 | 1.3 | 0 | 4/95 | 0.9 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 95.0 | 93.26 | 95 | 95/95 | 193.95 | 100 | 0.002 |
| 1990000 | 86.3 | 91.58 | 8 | 95/95 | 175.31 | 90 | 0.002 |
| 1991000 | 94.7 | 93.2 | 92 | 95/95 | 183.702 | 90 | 0.002 |
| 1992000 | 77.5 | 89.7 | 8 | 95/95 | 146.612 | 70 | 0.002 |
| 1993000 | 86.2 | 87.94 | 10 | 95/95 | 165.255 | 80 | 0.002 |
| 1994000 | 78.0 | 84.54 | 10 | 95/95 | 157.067 | 80 | 0.002 |
| 1995000 | 95.0 | 86.28 | 95 | 95/95 | 193.955 | 100 | 0.002 |
| 1996000 | 95.0 | 86.34 | 95 | 95/95 | 193.953 | 100 | 0.002 |
| 1997000 | 86.5 | 88.14 | 13 | 95/95 | 165.556 | 80 | 0.002 |
| 1998000 | 94.9 | 89.88 | 94 | 95/95 | 183.457 | 90 | 0.002 |
| 1999000 | 94.8 | 93.24 | 93 | 95/95 | 183.804 | 90 | 0.002 |
| 2000000 | 77.6 | 89.76 | 5 | 95/95 | 146.26 | 70 | 0.002 |
