# b47d-vecrepro-seed4

![b47d-vecrepro-seed4 progress](b47d-vecrepro-seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 91.88, perfect games 84%.

## Config

| setting | value |
|---|---|
| policy_name | b47d-vecrepro-seed4 |
| seed | 4 |
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
| eval | 100 episodes every 1000 steps, engine vec, 100 lanes in-process, no worker processes |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.0 |
| CHASE_SAFE_SHAPING | c=0.1, potential-based on head/food/tail in one region, gated to snake length >= 75 |
| FREE_SPACE_SHAPING | off |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2001 evals so far. Full series in [`b47d-vecrepro-seed4_evals.json`](b47d-vecrepro-seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.05 | 0.05 | 0 | 1/95 | -4.95 | 0 | 0.4 |
| 1000 | 0.87 | 0.87 | 0 | 8/95 | 0.37 | 0 | 0.4 |
| 2000 | 7.17 | 4.02 | 1 | 92/95 | 6.625 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 92.62 | 93.63 | 23 | 95/95 | 167.244 | 76 | 0.002 |
| 1990000 | 94.62 | 93.98 | 78 | 95/95 | 187.377 | 94 | 0.002 |
| 1991000 | 93.45 | 93.86 | 3 | 95/95 | 176.307 | 84 | 0.002 |
| 1992000 | 92.79 | 93.5 | 9 | 95/95 | 173.43 | 82 | 0.002 |
| 1993000 | 93.85 | 93.47 | 55 | 95/95 | 179.734 | 87 | 0.002 |
| 1994000 | 92.53 | 93.45 | 9 | 95/95 | 177.195 | 86 | 0.002 |
| 1995000 | 93.91 | 93.31 | 74 | 95/95 | 177.803 | 85 | 0.002 |
| 1996000 | 93.87 | 93.39 | 64 | 95/95 | 179.709 | 87 | 0.002 |
| 1997000 | 92.56 | 93.34 | 10 | 95/95 | 174.239 | 83 | 0.002 |
| 1998000 | 92.66 | 93.11 | 10 | 95/95 | 172.439 | 81 | 0.002 |
| 1999000 | 93.66 | 93.33 | 58 | 95/95 | 176.377 | 84 | 0.002 |
| 2000000 | 91.88 | 92.93 | 7 | 95/95 | 174.553 | 84 | 0.002 |
