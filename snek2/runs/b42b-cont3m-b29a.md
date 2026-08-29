# b42b-cont3m-b29a

![b42b-cont3m-b29a progress](b42b-cont3m-b29a.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1768000, avg score 94.7, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b42b-cont3m-b29a |
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
| max_steps | 3000000 |
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

422 evals so far. Full series in [`b42b-cont3m-b29a_evals.json`](b42b-cont3m-b29a_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1347000 | 95.0 | 95.0 | 95 | 95/95 | 193.954 | 100 | 0.002 |
| 1348000 | 94.8 | 94.8 | 93 | 95/95 | 183.806 | 90 | 0.002 |
| 1349000 | 95.0 | 94.9 | 95 | 95/95 | 193.956 | 100 | 0.002 |
| ... | | | | | | | |
| 1757000 | 95.0 | 94.92 | 95 | 95/95 | 193.951 | 100 | 0.002 |
| 1758000 | 83.2 | 92.64 | 6 | 95/95 | 162.262 | 80 | 0.002 |
| 1759000 | 94.4 | 92.52 | 91 | 95/95 | 173.452 | 80 | 0.002 |
| 1760000 | 95.0 | 92.52 | 95 | 95/95 | 193.956 | 100 | 0.002 |
| 1761000 | 94.3 | 92.38 | 90 | 95/95 | 173.353 | 80 | 0.002 |
| 1762000 | 88.7 | 91.12 | 32 | 95/95 | 177.712 | 90 | 0.002 |
| 1763000 | 94.7 | 93.42 | 92 | 95/95 | 183.702 | 90 | 0.002 |
| 1764000 | 95.0 | 93.54 | 95 | 95/95 | 193.955 | 100 | 0.002 |
| 1765000 | 95.0 | 93.54 | 95 | 95/95 | 193.95 | 100 | 0.002 |
| 1766000 | 85.8 | 91.84 | 3 | 95/95 | 174.813 | 90 | 0.002 |
| 1767000 | 94.8 | 93.06 | 93 | 95/95 | 183.801 | 90 | 0.002 |
| 1768000 | 94.7 | 93.06 | 92 | 95/95 | 183.706 | 90 | 0.002 |
