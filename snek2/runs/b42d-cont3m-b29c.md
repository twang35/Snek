# b42d-cont3m-b29c

![b42d-cont3m-b29c progress](b42d-cont3m-b29c.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1806000, avg score 94.5, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b42d-cont3m-b29c |
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

411 evals so far. Full series in [`b42d-cont3m-b29c_evals.json`](b42d-cont3m-b29c_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1396000 | 95.0 | 95.0 | 95 | 95/95 | 193.952 | 100 | 0.002 |
| 1397000 | 95.0 | 95.0 | 95 | 95/95 | 193.955 | 100 | 0.002 |
| 1398000 | 95.0 | 95.0 | 95 | 95/95 | 193.951 | 100 | 0.002 |
| ... | | | | | | | |
| 1795000 | 95.0 | 94.94 | 95 | 95/95 | 193.954 | 100 | 0.002 |
| 1796000 | 94.5 | 94.9 | 92 | 95/95 | 173.555 | 80 | 0.002 |
| 1797000 | 92.9 | 94.48 | 74 | 95/95 | 181.903 | 90 | 0.002 |
| 1798000 | 92.8 | 94.04 | 82 | 95/95 | 171.848 | 80 | 0.002 |
| 1799000 | 94.8 | 94.0 | 93 | 95/95 | 183.804 | 90 | 0.002 |
| 1800000 | 95.0 | 94.0 | 95 | 95/95 | 193.952 | 100 | 0.002 |
| 1801000 | 95.0 | 94.1 | 95 | 95/95 | 193.953 | 100 | 0.002 |
| 1802000 | 92.0 | 93.92 | 65 | 95/95 | 181.004 | 90 | 0.002 |
| 1803000 | 91.3 | 93.62 | 74 | 95/95 | 170.357 | 80 | 0.002 |
| 1804000 | 94.4 | 93.54 | 92 | 95/95 | 173.454 | 80 | 0.002 |
| 1805000 | 94.5 | 93.44 | 92 | 95/95 | 173.554 | 80 | 0.002 |
| 1806000 | 94.5 | 93.34 | 90 | 95/95 | 183.503 | 90 | 0.002 |
