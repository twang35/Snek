# b29d-chase10g75seed4

![b29d-chase10g75seed4 progress](b29d-chase10g75seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 93.8, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | b29d-chase10g75seed4 |
| seed | 4 |
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

2001 evals so far. Full series in [`b29d-chase10g75seed4_evals.json`](b29d-chase10g75seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 1.8 | 1.8 | 0 | 11/95 | 1.3 | 0 | 0.4 |
| 2000 | 8.1 | 4.95 | 1 | 53/95 | 7.6 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 93.9 | 92.7 | 90 | 95/95 | 163.0 | 70 | 0.002 |
| 1990000 | 87.1 | 91.16 | 18 | 95/95 | 166.153 | 80 | 0.002 |
| 1991000 | 93.5 | 90.9 | 80 | 95/95 | 182.498 | 90 | 0.002 |
| 1992000 | 94.7 | 90.94 | 92 | 95/95 | 183.702 | 90 | 0.002 |
| 1993000 | 94.1 | 92.66 | 92 | 95/95 | 163.203 | 70 | 0.002 |
| 1994000 | 94.1 | 92.7 | 86 | 95/95 | 183.097 | 90 | 0.002 |
| 1995000 | 90.4 | 93.36 | 70 | 95/95 | 169.455 | 80 | 0.002 |
| 1996000 | 94.3 | 93.52 | 88 | 95/95 | 183.303 | 90 | 0.002 |
| 1997000 | 93.9 | 93.36 | 90 | 95/95 | 163.01 | 70 | 0.002 |
| 1998000 | 92.6 | 93.06 | 74 | 95/95 | 171.651 | 80 | 0.002 |
| 1999000 | 93.1 | 92.86 | 85 | 95/95 | 172.156 | 80 | 0.002 |
| 2000000 | 93.8 | 93.54 | 90 | 95/95 | 152.502 | 60 | 0.002 |
