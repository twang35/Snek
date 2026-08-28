# b47c-vecrepro-seed3

![b47c-vecrepro-seed3 progress](b47c-vecrepro-seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 93.68, perfect games 82%.

## Config

| setting | value |
|---|---|
| policy_name | b47c-vecrepro-seed3 |
| seed | 3 |
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

2001 evals so far. Full series in [`b47c-vecrepro-seed3_evals.json`](b47c-vecrepro-seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.02 | 0.02 | 0 | 1/95 | -4.98 | 0 | 0.4 |
| 1000 | 1.0 | 1.0 | 0 | 9/95 | 0.5 | 0 | 0.4 |
| 2000 | 1.23 | 1.11 | 0 | 6/95 | 0.73 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 94.69 | 94.11 | 84 | 95/95 | 187.672 | 94 | 0.002 |
| 1990000 | 94.4 | 94.11 | 68 | 95/95 | 188.38 | 95 | 0.002 |
| 1991000 | 94.28 | 94.13 | 76 | 95/95 | 182.288 | 89 | 0.002 |
| 1992000 | 94.55 | 94.2 | 72 | 95/95 | 186.537 | 93 | 0.002 |
| 1993000 | 94.67 | 94.52 | 79 | 95/95 | 189.642 | 96 | 0.002 |
| 1994000 | 94.75 | 94.53 | 90 | 95/95 | 185.743 | 92 | 0.002 |
| 1995000 | 93.41 | 94.33 | 32 | 95/95 | 180.425 | 88 | 0.002 |
| 1996000 | 94.5 | 94.38 | 86 | 95/95 | 180.521 | 87 | 0.002 |
| 1997000 | 93.81 | 94.23 | 44 | 95/95 | 177.748 | 85 | 0.002 |
| 1998000 | 93.53 | 94.0 | 4 | 95/95 | 183.529 | 91 | 0.002 |
| 1999000 | 94.65 | 93.98 | 84 | 95/95 | 186.637 | 93 | 0.002 |
| 2000000 | 93.68 | 94.03 | 71 | 95/95 | 174.723 | 82 | 0.002 |
