# b29c-chase10g75seed3

![b29c-chase10g75seed3 progress](b29c-chase10g75seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 95.0, perfect games 100%.

## Config

| setting | value |
|---|---|
| policy_name | b29c-chase10g75seed3 |
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
| CHASE_SAFE_SHAPING | c=0.1, potential-based on head/food/tail in one region, gated to snake length >= 75 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2001 evals so far. Full series in [`b29c-chase10g75seed3_evals.json`](b29c-chase10g75seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 4.6 | 4.6 | 0 | 12/95 | 4.1 | 0 | 0.4 |
| 2000 | 1.5 | 3.05 | 0 | 7/95 | 1.0 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 94.8 | 94.62 | 93 | 95/95 | 183.804 | 90 | 0.002 |
| 1990000 | 92.2 | 94.16 | 68 | 95/95 | 170.811 | 80 | 0.002 |
| 1991000 | 95.0 | 94.24 | 95 | 95/95 | 193.956 | 100 | 0.002 |
| 1992000 | 94.7 | 94.18 | 92 | 95/95 | 183.705 | 90 | 0.002 |
| 1993000 | 95.0 | 94.34 | 95 | 95/95 | 193.951 | 100 | 0.002 |
| 1994000 | 92.0 | 93.78 | 73 | 95/95 | 161.107 | 70 | 0.002 |
| 1995000 | 94.1 | 94.16 | 89 | 95/95 | 173.153 | 80 | 0.002 |
| 1996000 | 95.0 | 94.16 | 95 | 95/95 | 193.955 | 100 | 0.002 |
| 1997000 | 93.6 | 93.94 | 82 | 95/95 | 172.205 | 80 | 0.002 |
| 1998000 | 94.5 | 93.84 | 90 | 95/95 | 183.505 | 90 | 0.002 |
| 1999000 | 94.9 | 94.42 | 94 | 95/95 | 183.451 | 90 | 0.002 |
| 2000000 | 95.0 | 94.6 | 95 | 95/95 | 193.955 | 100 | 0.002 |
