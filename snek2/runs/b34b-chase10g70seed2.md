# b34b-chase10g70seed2

![b34b-chase10g70seed2 progress](b34b-chase10g70seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 95.0, perfect games 100%.

## Config

| setting | value |
|---|---|
| policy_name | b34b-chase10g70seed2 |
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
| CHASE_SAFE_SHAPING | c=0.1, potential-based on head/food/tail in one region, gated to snake length >= 70 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2001 evals so far. Full series in [`b34b-chase10g70seed2_evals.json`](b34b-chase10g70seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -3.2 | 0 | 0.4 |
| 1000 | 0.6 | 0.6 | 0 | 2/95 | 0.1 | 0 | 0.4 |
| 2000 | 1.2 | 0.9 | 0 | 3/95 | 0.7 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 94.3 | 88.94 | 90 | 95/95 | 173.331 | 80 | 0.002 |
| 1990000 | 94.5 | 88.92 | 92 | 95/95 | 173.528 | 80 | 0.002 |
| 1991000 | 92.9 | 92.42 | 84 | 95/95 | 171.931 | 80 | 0.002 |
| 1992000 | 82.9 | 91.86 | 2 | 95/95 | 132.086 | 50 | 0.002 |
| 1993000 | 85.0 | 89.92 | 0 | 95/95 | 154.083 | 70 | 0.002 |
| 1994000 | 84.8 | 88.02 | 2 | 95/95 | 163.84 | 80 | 0.002 |
| 1995000 | 94.7 | 88.06 | 92 | 95/95 | 183.683 | 90 | 0.002 |
| 1996000 | 95.0 | 88.48 | 95 | 95/95 | 193.93 | 100 | 0.002 |
| 1997000 | 85.6 | 89.02 | 1 | 95/95 | 174.588 | 90 | 0.002 |
| 1998000 | 95.0 | 91.02 | 95 | 95/95 | 193.936 | 100 | 0.002 |
| 1999000 | 95.0 | 93.06 | 95 | 95/95 | 193.936 | 100 | 0.002 |
| 2000000 | 95.0 | 93.12 | 95 | 95/95 | 193.935 | 100 | 0.002 |
