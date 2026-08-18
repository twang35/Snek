# b37b-chase10g75seed6

![b37b-chase10g75seed6 progress](b37b-chase10g75seed6.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 92.8, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b37b-chase10g75seed6 |
| seed | 6 |
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

2001 evals so far. Full series in [`b37b-chase10g75seed6_evals.json`](b37b-chase10g75seed6_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -1.3 | 0 | 0.4 |
| 1000 | 0.5 | 0.5 | 0 | 1/95 | 0.0 | 0 | 0.4 |
| 2000 | 0.5 | 0.5 | 0 | 3/95 | 0.0 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 95.0 | 94.06 | 95 | 95/95 | 193.955 | 100 | 0.002 |
| 1990000 | 86.2 | 92.36 | 7 | 95/95 | 175.201 | 90 | 0.002 |
| 1991000 | 94.7 | 92.36 | 92 | 95/95 | 183.704 | 90 | 0.002 |
| 1992000 | 95.0 | 92.66 | 95 | 95/95 | 193.947 | 100 | 0.002 |
| 1993000 | 95.0 | 93.18 | 95 | 95/95 | 193.95 | 100 | 0.002 |
| 1994000 | 95.0 | 93.18 | 95 | 95/95 | 193.954 | 100 | 0.002 |
| 1995000 | 84.6 | 92.86 | 21 | 95/95 | 153.706 | 70 | 0.002 |
| 1996000 | 91.2 | 92.16 | 75 | 95/95 | 160.304 | 70 | 0.002 |
| 1997000 | 95.0 | 92.16 | 95 | 95/95 | 193.952 | 100 | 0.002 |
| 1998000 | 95.0 | 92.16 | 95 | 95/95 | 193.952 | 100 | 0.002 |
| 1999000 | 93.3 | 91.82 | 80 | 95/95 | 172.354 | 80 | 0.002 |
| 2000000 | 92.8 | 93.46 | 73 | 95/95 | 181.795 | 90 | 0.002 |
