# b41a-b29repro-seed1

![b41a-b29repro-seed1 progress](b41a-b29repro-seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 94.6, perfect games 80%.

## Config

| setting | value |
|---|---|
| policy_name | b41a-b29repro-seed1 |
| seed | 1 |
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

2001 evals so far. Full series in [`b41a-b29repro-seed1_evals.json`](b41a-b29repro-seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 4.7 | 4.7 | 0 | 18/95 | 4.2 | 0 | 0.4 |
| 2000 | 1.0 | 2.85 | 0 | 3/95 | 0.5 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 95.0 | 91.74 | 95 | 95/95 | 193.951 | 100 | 0.0022 |
| 1990000 | 94.8 | 92.58 | 93 | 95/95 | 183.803 | 90 | 0.0022 |
| 1991000 | 94.8 | 94.34 | 93 | 95/95 | 183.349 | 90 | 0.0022 |
| 1992000 | 76.7 | 91.18 | 5 | 95/95 | 125.464 | 50 | 0.0022 |
| 1993000 | 82.9 | 88.84 | 6 | 95/95 | 152.018 | 70 | 0.0022 |
| 1994000 | 85.3 | 86.9 | 7 | 95/95 | 144.011 | 60 | 0.0023 |
| 1995000 | 94.8 | 86.9 | 93 | 95/95 | 183.348 | 90 | 0.0022 |
| 1996000 | 85.2 | 84.98 | 5 | 95/95 | 134.409 | 50 | 0.0023 |
| 1997000 | 94.8 | 88.6 | 93 | 95/95 | 183.801 | 90 | 0.0022 |
| 1998000 | 82.9 | 88.6 | 5 | 95/95 | 142.055 | 60 | 0.0022 |
| 1999000 | 86.1 | 88.76 | 9 | 95/95 | 165.16 | 80 | 0.0022 |
| 2000000 | 94.6 | 88.72 | 93 | 95/95 | 172.745 | 80 | 0.0022 |
