# b34a-chase10g70seed1

![b34a-chase10g70seed1 progress](b34a-chase10g70seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 94.7, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b34a-chase10g70seed1 |
| seed | 1 |
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

2001 evals so far. Full series in [`b34a-chase10g70seed1_evals.json`](b34a-chase10g70seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 1.0 | 1.0 | 0 | 5/95 | 0.5 | 0 | 0.4 |
| 2000 | 0.8 | 0.9 | 0 | 3/95 | 0.3 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 93.5 | 93.5 | 84 | 95/95 | 172.528 | 80 | 0.0023 |
| 1990000 | 91.0 | 93.04 | 76 | 95/95 | 149.691 | 60 | 0.0024 |
| 1991000 | 94.0 | 92.96 | 90 | 95/95 | 163.071 | 70 | 0.0024 |
| 1992000 | 94.8 | 93.38 | 93 | 95/95 | 183.784 | 90 | 0.0023 |
| 1993000 | 93.6 | 93.38 | 86 | 95/95 | 162.684 | 70 | 0.0023 |
| 1994000 | 95.0 | 93.68 | 95 | 95/95 | 193.932 | 100 | 0.0022 |
| 1995000 | 93.5 | 94.18 | 86 | 95/95 | 152.635 | 60 | 0.0023 |
| 1996000 | 95.0 | 94.38 | 95 | 95/95 | 193.928 | 100 | 0.0022 |
| 1997000 | 90.9 | 93.6 | 72 | 95/95 | 159.536 | 70 | 0.0022 |
| 1998000 | 92.5 | 93.38 | 72 | 95/95 | 171.083 | 80 | 0.0022 |
| 1999000 | 93.1 | 93.0 | 82 | 95/95 | 151.784 | 60 | 0.0023 |
| 2000000 | 94.7 | 93.24 | 92 | 95/95 | 183.683 | 90 | 0.0022 |
