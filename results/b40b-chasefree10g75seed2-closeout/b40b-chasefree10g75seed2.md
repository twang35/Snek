# b40b-chasefree10g75seed2

![b40b-chasefree10g75seed2 progress](b40b-chasefree10g75seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 95.0, perfect games 100%.

## Config

| setting | value |
|---|---|
| policy_name | b40b-chasefree10g75seed2 |
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
| FREE_SPACE_SHAPING | c=0.1, potential-based on 1/open-region-count, gated to snake length >= 75 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2001 evals so far. Full series in [`b40b-chasefree10g75seed2_evals.json`](b40b-chasefree10g75seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -3.2 | 0 | 0.4 |
| 1000 | 0.6 | 0.6 | 0 | 2/95 | 0.1 | 0 | 0.4 |
| 2000 | 0.6 | 0.6 | 0 | 3/95 | 0.1 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 94.7 | 94.78 | 92 | 95/95 | 183.644 | 90 | 0.002 |
| 1990000 | 93.9 | 94.66 | 89 | 95/95 | 162.947 | 70 | 0.002 |
| 1991000 | 93.3 | 94.32 | 84 | 95/95 | 162.335 | 70 | 0.002 |
| 1992000 | 93.4 | 94.0 | 88 | 95/95 | 162.437 | 70 | 0.002 |
| 1993000 | 93.7 | 93.8 | 84 | 95/95 | 172.663 | 80 | 0.002 |
| 1994000 | 94.5 | 93.76 | 92 | 95/95 | 173.485 | 80 | 0.002 |
| 1995000 | 93.1 | 93.6 | 83 | 95/95 | 162.142 | 70 | 0.002 |
| 1996000 | 94.7 | 93.88 | 92 | 95/95 | 183.636 | 90 | 0.002 |
| 1997000 | 92.7 | 93.74 | 72 | 95/95 | 181.653 | 90 | 0.002 |
| 1998000 | 95.0 | 94.0 | 95 | 95/95 | 193.889 | 100 | 0.002 |
| 1999000 | 92.7 | 93.64 | 72 | 95/95 | 181.657 | 90 | 0.002 |
| 2000000 | 95.0 | 94.02 | 95 | 95/95 | 193.904 | 100 | 0.002 |
