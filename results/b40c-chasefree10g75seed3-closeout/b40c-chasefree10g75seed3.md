# b40c-chasefree10g75seed3

![b40c-chasefree10g75seed3 progress](b40c-chasefree10g75seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 92.7, perfect games 70%.

## Config

| setting | value |
|---|---|
| policy_name | b40c-chasefree10g75seed3 |
| seed | 3 |
| zeroed_observations | none |
| learning_rate | 1e-05 |
| adam_epsilon | 1e-07 |
| perfect_game_reward | 100.0 |
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
| algo | ddqn, scalar head, Huber TD error |
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
| FREE_SPACE_SHAPING | c=0.1, potential-based on 1/open-region-count, gated to snake length >= 75 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2001 evals so far. Full series in [`b40c-chasefree10g75seed3_evals.json`](b40c-chasefree10g75seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 1.5 | 1.5 | 0 | 4/95 | 1.0 | 0 | 0.4 |
| 2000 | 1.1 | 1.3 | 0 | 6/95 | 0.6 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 93.3 | 93.8 | 82 | 95/95 | 161.427 | 70 | 0.002 |
| 1990000 | 94.7 | 94.08 | 92 | 95/95 | 183.184 | 90 | 0.002 |
| 1991000 | 94.9 | 94.08 | 94 | 95/95 | 183.358 | 90 | 0.002 |
| 1992000 | 89.6 | 93.44 | 41 | 95/95 | 178.557 | 90 | 0.002 |
| 1993000 | 94.7 | 93.44 | 92 | 95/95 | 183.633 | 90 | 0.002 |
| 1994000 | 94.7 | 93.72 | 92 | 95/95 | 183.646 | 90 | 0.002 |
| 1995000 | 94.9 | 93.76 | 94 | 95/95 | 183.403 | 90 | 0.002 |
| 1996000 | 94.7 | 93.72 | 92 | 95/95 | 183.659 | 90 | 0.002 |
| 1997000 | 94.3 | 94.66 | 88 | 95/95 | 183.235 | 90 | 0.002 |
| 1998000 | 94.8 | 94.68 | 93 | 95/95 | 183.733 | 90 | 0.002 |
| 1999000 | 95.0 | 94.74 | 95 | 95/95 | 193.873 | 100 | 0.002 |
| 2000000 | 92.7 | 94.3 | 80 | 95/95 | 161.282 | 70 | 0.002 |
