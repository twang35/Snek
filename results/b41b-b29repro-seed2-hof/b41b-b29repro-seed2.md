# b41b-b29repro-seed2

![b41b-b29repro-seed2 progress](b41b-b29repro-seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 87.3, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b41b-b29repro-seed2 |
| seed | 2 |
| zeroed_observations | none |
| learning_rate | 1e-05 |
| adam_epsilon | 1e-07 |
| perfect_game_reward | 100.0 |
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
| algo | ddqn, scalar head, Huber TD error |
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
| FREE_SPACE_SHAPING | off |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2001 evals so far. Full series in [`b41b-b29repro-seed2_evals.json`](b41b-b29repro-seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -3.2 | 0 | 0.4 |
| 1000 | 1.5 | 1.5 | 0 | 5/95 | 1.0 | 0 | 0.4 |
| 2000 | 0.9 | 1.2 | 0 | 3/95 | 0.4 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 90.3 | 90.8 | 50 | 95/95 | 169.356 | 80 | 0.002 |
| 1990000 | 95.0 | 90.8 | 95 | 95/95 | 193.954 | 100 | 0.002 |
| 1991000 | 95.0 | 90.94 | 95 | 95/95 | 193.952 | 100 | 0.002 |
| 1992000 | 95.0 | 94.06 | 95 | 95/95 | 193.949 | 100 | 0.002 |
| 1993000 | 95.0 | 94.06 | 95 | 95/95 | 193.955 | 100 | 0.002 |
| 1994000 | 93.7 | 94.74 | 86 | 95/95 | 162.349 | 70 | 0.002 |
| 1995000 | 92.7 | 94.28 | 84 | 95/95 | 161.803 | 70 | 0.002 |
| 1996000 | 95.0 | 94.28 | 95 | 95/95 | 193.952 | 100 | 0.002 |
| 1997000 | 93.5 | 93.98 | 82 | 95/95 | 172.554 | 80 | 0.002 |
| 1998000 | 86.6 | 92.3 | 11 | 95/95 | 175.605 | 90 | 0.002 |
| 1999000 | 94.3 | 92.42 | 88 | 95/95 | 183.299 | 90 | 0.002 |
| 2000000 | 87.3 | 91.34 | 18 | 95/95 | 176.307 | 90 | 0.002 |
