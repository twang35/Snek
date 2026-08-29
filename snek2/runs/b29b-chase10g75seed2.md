# b29b-chase10g75seed2

![b29b-chase10g75seed2 progress](b29b-chase10g75seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 95.0, perfect games 100%.

## Config

| setting | value |
|---|---|
| policy_name | b29b-chase10g75seed2 |
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
| CHASE_SAFE_SHAPING | c=0.1, potential-based on head/food/tail in one region, gated to snake length >= 75 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2001 evals so far. Full series in [`b29b-chase10g75seed2_evals.json`](b29b-chase10g75seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -3.2 | 0 | 0.4 |
| 1000 | 0.6 | 0.6 | 0 | 2/95 | 0.1 | 0 | 0.4 |
| 2000 | 1.2 | 0.9 | 0 | 3/95 | 0.7 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 94.6 | 88.4 | 91 | 95/95 | 183.609 | 90 | 0.002 |
| 1990000 | 77.2 | 85.44 | 5 | 95/95 | 156.264 | 80 | 0.002 |
| 1991000 | 86.0 | 85.54 | 5 | 95/95 | 175.007 | 90 | 0.002 |
| 1992000 | 95.0 | 89.18 | 95 | 95/95 | 193.951 | 100 | 0.002 |
| 1993000 | 85.8 | 87.72 | 6 | 95/95 | 164.858 | 80 | 0.002 |
| 1994000 | 85.7 | 85.94 | 5 | 95/95 | 164.758 | 80 | 0.002 |
| 1995000 | 94.8 | 89.46 | 93 | 95/95 | 183.803 | 90 | 0.002 |
| 1996000 | 85.3 | 89.32 | 1 | 95/95 | 164.357 | 80 | 0.002 |
| 1997000 | 76.5 | 85.62 | 2 | 95/95 | 155.567 | 80 | 0.002 |
| 1998000 | 86.0 | 85.66 | 5 | 95/95 | 175.01 | 90 | 0.002 |
| 1999000 | 95.0 | 87.52 | 95 | 95/95 | 193.954 | 100 | 0.002 |
| 2000000 | 95.0 | 87.56 | 95 | 95/95 | 193.952 | 100 | 0.002 |
