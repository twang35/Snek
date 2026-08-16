# b28a-chase20g85seed1

![b28a-chase20g85seed1 progress](b28a-chase20g85seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 95.0, perfect games 100%.

## Config

| setting | value |
|---|---|
| policy_name | b28a-chase20g85seed1 |
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
| CHASE_SAFE_SHAPING | c=0.2, potential-based on head/food/tail in one region, gated to snake length >= 85 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2001 evals so far. Full series in [`b28a-chase20g85seed1_evals.json`](b28a-chase20g85seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 3.7 | 3.7 | 0 | 13/95 | 3.2 | 0 | 0.4 |
| 2000 | 1.3 | 2.5 | 0 | 3/95 | 0.8 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 94.7 | 94.68 | 92 | 95/95 | 183.714 | 90 | 0.002 |
| 1990000 | 94.8 | 94.7 | 93 | 95/95 | 183.811 | 90 | 0.002 |
| 1991000 | 95.0 | 94.7 | 95 | 95/95 | 193.968 | 100 | 0.002 |
| 1992000 | 94.7 | 94.7 | 92 | 95/95 | 183.712 | 90 | 0.002 |
| 1993000 | 93.3 | 94.5 | 78 | 95/95 | 182.317 | 90 | 0.002 |
| 1994000 | 94.7 | 94.5 | 92 | 95/95 | 183.712 | 90 | 0.002 |
| 1995000 | 95.0 | 94.54 | 95 | 95/95 | 193.966 | 100 | 0.002 |
| 1996000 | 94.7 | 94.48 | 92 | 95/95 | 183.709 | 90 | 0.002 |
| 1997000 | 94.7 | 94.48 | 92 | 95/95 | 183.716 | 90 | 0.002 |
| 1998000 | 94.8 | 94.78 | 93 | 95/95 | 183.814 | 90 | 0.002 |
| 1999000 | 95.0 | 94.84 | 95 | 95/95 | 193.967 | 100 | 0.002 |
| 2000000 | 95.0 | 94.84 | 95 | 95/95 | 193.964 | 100 | 0.002 |
