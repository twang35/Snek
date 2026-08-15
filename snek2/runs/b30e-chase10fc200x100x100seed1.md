# b30e-chase10fc200x100x100seed1

![b30e-chase10fc200x100x100seed1 progress](b30e-chase10fc200x100x100seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 93.9, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | b30e-chase10fc200x100x100seed1 |
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

2001 evals so far. Full series in [`b30e-chase10fc200x100x100seed1_evals.json`](b30e-chase10fc200x100x100seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 1.4 | 1.4 | 0 | 3/95 | 0.9 | 0 | 0.4 |
| 2000 | 2.0 | 1.7 | 0 | 6/95 | 1.5 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 95.0 | 92.52 | 95 | 95/95 | 193.984 | 100 | 0.002 |
| 1990000 | 85.3 | 92.54 | 2 | 95/95 | 153.53 | 70 | 0.002 |
| 1991000 | 94.4 | 92.5 | 93 | 95/95 | 162.18 | 70 | 0.0021 |
| 1992000 | 75.3 | 88.74 | 0 | 95/95 | 123.633 | 50 | 0.0021 |
| 1993000 | 94.5 | 88.9 | 92 | 95/95 | 173.132 | 80 | 0.0021 |
| 1994000 | 94.6 | 88.82 | 93 | 95/95 | 172.781 | 80 | 0.0021 |
| 1995000 | 85.2 | 88.8 | 2 | 95/95 | 153.882 | 70 | 0.0021 |
| 1996000 | 94.1 | 88.74 | 90 | 95/95 | 162.329 | 70 | 0.0021 |
| 1997000 | 85.5 | 90.78 | 2 | 95/95 | 164.129 | 80 | 0.0022 |
| 1998000 | 92.9 | 90.46 | 74 | 95/95 | 181.93 | 90 | 0.0021 |
| 1999000 | 93.0 | 90.14 | 89 | 95/95 | 142.225 | 50 | 0.0022 |
| 2000000 | 93.9 | 91.88 | 90 | 95/95 | 151.732 | 60 | 0.0022 |
