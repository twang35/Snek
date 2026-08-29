# b30f-chase10fc200x100x100seed2

![b30f-chase10fc200x100x100seed2 progress](b30f-chase10fc200x100x100seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 94.3, perfect games 80%.

## Config

| setting | value |
|---|---|
| policy_name | b30f-chase10fc200x100x100seed2 |
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

2001 evals so far. Full series in [`b30f-chase10fc200x100x100seed2_evals.json`](b30f-chase10fc200x100x100seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.9 | 0.9 | 0 | 3/95 | 0.4 | 0 | 0.4 |
| 2000 | 0.8 | 0.85 | 0 | 3/95 | 0.3 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 93.4 | 93.1 | 83 | 95/95 | 161.63 | 70 | 0.002 |
| 1990000 | 94.5 | 93.42 | 92 | 95/95 | 173.133 | 80 | 0.002 |
| 1991000 | 93.5 | 94.16 | 87 | 95/95 | 151.328 | 60 | 0.002 |
| 1992000 | 94.6 | 94.16 | 93 | 95/95 | 172.778 | 80 | 0.002 |
| 1993000 | 94.6 | 94.12 | 93 | 95/95 | 172.778 | 80 | 0.002 |
| 1994000 | 93.5 | 94.14 | 88 | 95/95 | 140.929 | 50 | 0.0021 |
| 1995000 | 94.1 | 94.06 | 88 | 95/95 | 172.729 | 80 | 0.0021 |
| 1996000 | 94.8 | 94.32 | 93 | 95/95 | 183.382 | 90 | 0.0021 |
| 1997000 | 95.0 | 94.4 | 95 | 95/95 | 193.977 | 100 | 0.0021 |
| 1998000 | 94.7 | 94.42 | 92 | 95/95 | 183.729 | 90 | 0.002 |
| 1999000 | 95.0 | 94.72 | 95 | 95/95 | 193.979 | 100 | 0.002 |
| 2000000 | 94.3 | 94.76 | 90 | 95/95 | 172.926 | 80 | 0.002 |
