# b27g-chase10g85seed3

![b27g-chase10g85seed3 progress](b27g-chase10g85seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 94.8, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b27g-chase10g85seed3 |
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
| CHASE_SAFE_SHAPING | c=0.1, potential-based on head/food/tail in one region, gated to snake length >= 85 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2001 evals so far. Full series in [`b27g-chase10g85seed3_evals.json`](b27g-chase10g85seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 1.4 | 1.4 | 0 | 6/95 | 0.9 | 0 | 0.4 |
| 2000 | 3.9 | 2.65 | 0 | 13/95 | 3.4 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 95.0 | 94.96 | 95 | 95/95 | 193.98 | 100 | 0.002 |
| 1990000 | 95.0 | 94.96 | 95 | 95/95 | 193.983 | 100 | 0.002 |
| 1991000 | 92.8 | 94.52 | 73 | 95/95 | 181.835 | 90 | 0.002 |
| 1992000 | 94.5 | 94.42 | 90 | 95/95 | 183.53 | 90 | 0.002 |
| 1993000 | 95.0 | 94.46 | 95 | 95/95 | 193.983 | 100 | 0.002 |
| 1994000 | 94.8 | 94.42 | 93 | 95/95 | 183.832 | 90 | 0.002 |
| 1995000 | 92.7 | 93.96 | 72 | 95/95 | 181.736 | 90 | 0.002 |
| 1996000 | 92.9 | 93.98 | 74 | 95/95 | 181.935 | 90 | 0.002 |
| 1997000 | 92.2 | 93.52 | 67 | 95/95 | 181.234 | 90 | 0.002 |
| 1998000 | 95.0 | 93.52 | 95 | 95/95 | 193.983 | 100 | 0.002 |
| 1999000 | 92.2 | 93.0 | 67 | 95/95 | 181.234 | 90 | 0.002 |
| 2000000 | 94.8 | 93.42 | 93 | 95/95 | 183.383 | 90 | 0.002 |
