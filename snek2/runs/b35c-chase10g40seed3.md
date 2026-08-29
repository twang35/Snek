# b35c-chase10g40seed3

![b35c-chase10g40seed3 progress](b35c-chase10g40seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 93.3, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b35c-chase10g40seed3 |
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
| CHASE_SAFE_SHAPING | c=0.1, potential-based on head/food/tail in one region, gated to snake length >= 40 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2001 evals so far. Full series in [`b35c-chase10g40seed3_evals.json`](b35c-chase10g40seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 5.1 | 5.1 | 0 | 15/95 | 4.6 | 0 | 0.4 |
| 2000 | 1.4 | 3.25 | 0 | 4/95 | 0.9 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 95.0 | 94.3 | 95 | 95/95 | 193.808 | 100 | 0.002 |
| 1990000 | 85.9 | 92.48 | 4 | 95/95 | 174.783 | 90 | 0.002 |
| 1991000 | 95.0 | 92.54 | 95 | 95/95 | 193.82 | 100 | 0.002 |
| 1992000 | 95.0 | 92.6 | 95 | 95/95 | 193.806 | 100 | 0.002 |
| 1993000 | 89.5 | 92.08 | 40 | 95/95 | 178.375 | 90 | 0.002 |
| 1994000 | 85.9 | 90.26 | 4 | 95/95 | 174.792 | 90 | 0.002 |
| 1995000 | 95.0 | 92.08 | 95 | 95/95 | 193.805 | 100 | 0.002 |
| 1996000 | 93.1 | 91.7 | 76 | 95/95 | 181.96 | 90 | 0.002 |
| 1997000 | 95.0 | 91.7 | 95 | 95/95 | 193.814 | 100 | 0.002 |
| 1998000 | 93.8 | 92.56 | 83 | 95/95 | 182.665 | 90 | 0.002 |
| 1999000 | 95.0 | 94.38 | 95 | 95/95 | 193.806 | 100 | 0.002 |
| 2000000 | 93.3 | 94.04 | 78 | 95/95 | 182.14 | 90 | 0.002 |
