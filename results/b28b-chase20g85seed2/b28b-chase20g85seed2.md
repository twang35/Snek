# b28b-chase20g85seed2

![b28b-chase20g85seed2 progress](b28b-chase20g85seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 95.0, perfect games 100%.

## Config

| setting | value |
|---|---|
| policy_name | b28b-chase20g85seed2 |
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
| CHASE_SAFE_SHAPING | c=0.2, potential-based on head/food/tail in one region, gated to snake length >= 85 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2001 evals so far. Full series in [`b28b-chase20g85seed2_evals.json`](b28b-chase20g85seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -3.2 | 0 | 0.4 |
| 1000 | 0.6 | 0.6 | 0 | 2/95 | 0.1 | 0 | 0.4 |
| 2000 | 0.6 | 0.6 | 0 | 3/95 | 0.1 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 94.1 | 93.02 | 88 | 95/95 | 173.162 | 80 | 0.0024 |
| 1990000 | 90.2 | 93.22 | 59 | 95/95 | 138.966 | 50 | 0.0025 |
| 1991000 | 94.1 | 93.32 | 92 | 95/95 | 152.81 | 60 | 0.0025 |
| 1992000 | 94.7 | 93.5 | 93 | 95/95 | 173.312 | 80 | 0.0025 |
| 1993000 | 95.0 | 93.62 | 95 | 95/95 | 193.965 | 100 | 0.0024 |
| 1994000 | 94.3 | 93.66 | 92 | 95/95 | 162.963 | 70 | 0.0024 |
| 1995000 | 92.8 | 94.18 | 88 | 95/95 | 131.162 | 40 | 0.0024 |
| 1996000 | 93.9 | 94.14 | 88 | 95/95 | 172.966 | 80 | 0.0024 |
| 1997000 | 94.3 | 94.06 | 90 | 95/95 | 173.365 | 80 | 0.0024 |
| 1998000 | 92.8 | 93.62 | 82 | 95/95 | 161.916 | 70 | 0.0024 |
| 1999000 | 92.5 | 93.26 | 84 | 95/95 | 161.612 | 70 | 0.0024 |
| 2000000 | 95.0 | 93.7 | 95 | 95/95 | 193.968 | 100 | 0.0023 |
