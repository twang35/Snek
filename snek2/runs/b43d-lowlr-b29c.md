# b43d-lowlr-b29c

![b43d-lowlr-b29c progress](b43d-lowlr-b29c.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1537000, avg score 95.0, perfect games 100%.

## Config

| setting | value |
|---|---|
| policy_name | b43d-lowlr-b29c |
| seed | 3 |
| zeroed_observations | none |
| learning_rate | 1e-06 |
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

142 evals so far. Full series in [`b43d-lowlr-b29c_evals.json`](b43d-lowlr-b29c_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1396000 | 95.0 | 95.0 | 95 | 95/95 | 193.952 | 100 | 0.002 |
| 1397000 | 92.6 | 92.6 | 71 | 95/95 | 181.603 | 90 | 0.002 |
| 1398000 | 90.7 | 91.65 | 53 | 95/95 | 169.308 | 80 | 0.002 |
| ... | | | | | | | |
| 1526000 | 95.0 | 95.0 | 95 | 95/95 | 193.956 | 100 | 0.002 |
| 1527000 | 95.0 | 95.0 | 95 | 95/95 | 193.953 | 100 | 0.002 |
| 1528000 | 95.0 | 95.0 | 95 | 95/95 | 193.954 | 100 | 0.002 |
| 1529000 | 92.9 | 94.58 | 74 | 95/95 | 181.459 | 90 | 0.002 |
| 1530000 | 95.0 | 94.58 | 95 | 95/95 | 193.957 | 100 | 0.002 |
| 1531000 | 95.0 | 94.58 | 95 | 95/95 | 193.956 | 100 | 0.002 |
| 1532000 | 95.0 | 94.58 | 95 | 95/95 | 193.959 | 100 | 0.002 |
| 1533000 | 94.7 | 94.52 | 92 | 95/95 | 183.706 | 90 | 0.002 |
| 1534000 | 95.0 | 94.94 | 95 | 95/95 | 193.957 | 100 | 0.002 |
| 1535000 | 95.0 | 94.94 | 95 | 95/95 | 193.955 | 100 | 0.002 |
| 1536000 | 95.0 | 94.94 | 95 | 95/95 | 193.956 | 100 | 0.002 |
| 1537000 | 95.0 | 94.94 | 95 | 95/95 | 193.956 | 100 | 0.002 |
