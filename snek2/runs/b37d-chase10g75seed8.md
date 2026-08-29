# b37d-chase10g75seed8

![b37d-chase10g75seed8 progress](b37d-chase10g75seed8.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 94.0, perfect games 80%.

## Config

| setting | value |
|---|---|
| policy_name | b37d-chase10g75seed8 |
| seed | 8 |
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

2001 evals so far. Full series in [`b37d-chase10g75seed8_evals.json`](b37d-chase10g75seed8_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 1/95 | -0.1 | 0 | 0.4 |
| 2000 | 0.6 | 0.5 | 0 | 2/95 | 0.1 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 94.6 | 94.02 | 93 | 95/95 | 173.21 | 80 | 0.002 |
| 1990000 | 93.5 | 94.02 | 82 | 95/95 | 172.108 | 80 | 0.002 |
| 1991000 | 93.9 | 93.84 | 86 | 95/95 | 172.961 | 80 | 0.002 |
| 1992000 | 85.8 | 92.1 | 3 | 95/95 | 174.811 | 90 | 0.002 |
| 1993000 | 93.2 | 92.2 | 85 | 95/95 | 141.057 | 50 | 0.002 |
| 1994000 | 94.5 | 92.18 | 92 | 95/95 | 173.106 | 80 | 0.002 |
| 1995000 | 85.1 | 90.5 | 0 | 95/95 | 154.205 | 70 | 0.002 |
| 1996000 | 84.1 | 88.54 | 0 | 95/95 | 132.858 | 50 | 0.0021 |
| 1997000 | 94.3 | 90.24 | 92 | 95/95 | 162.959 | 70 | 0.0021 |
| 1998000 | 94.2 | 90.44 | 92 | 95/95 | 163.306 | 70 | 0.0021 |
| 1999000 | 94.5 | 90.44 | 92 | 95/95 | 173.108 | 80 | 0.0021 |
| 2000000 | 94.0 | 92.22 | 89 | 95/95 | 173.057 | 80 | 0.0021 |
