# b43c-lowlr-b40b

![b43c-lowlr-b40b progress](b43c-lowlr-b40b.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.5, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b43c-lowlr-b40b |
| seed | 2 |
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

1488 evals so far. Full series in [`b43c-lowlr-b40b_evals.json`](b43c-lowlr-b40b_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1513000 | 95.0 | 95.0 | 95 | 95/95 | 193.955 | 100 | 0.002 |
| 1514000 | 95.0 | 95.0 | 95 | 95/95 | 193.953 | 100 | 0.002 |
| 1515000 | 95.0 | 95.0 | 95 | 95/95 | 193.955 | 100 | 0.002 |
| ... | | | | | | | |
| 2989000 | 94.7 | 93.22 | 92 | 95/95 | 183.696 | 90 | 0.002 |
| 2990000 | 93.8 | 93.08 | 86 | 95/95 | 172.852 | 80 | 0.002 |
| 2991000 | 95.0 | 93.68 | 95 | 95/95 | 193.948 | 100 | 0.002 |
| 2992000 | 91.6 | 93.7 | 68 | 95/95 | 170.66 | 80 | 0.002 |
| 2993000 | 92.5 | 93.52 | 70 | 95/95 | 181.502 | 90 | 0.002 |
| 2994000 | 93.6 | 93.3 | 81 | 95/95 | 182.602 | 90 | 0.002 |
| 2995000 | 94.5 | 93.44 | 90 | 95/95 | 183.501 | 90 | 0.002 |
| 2996000 | 93.6 | 93.16 | 81 | 95/95 | 182.603 | 90 | 0.002 |
| 2997000 | 85.9 | 92.02 | 4 | 95/95 | 174.906 | 90 | 0.002 |
| 2998000 | 95.0 | 92.52 | 95 | 95/95 | 193.946 | 100 | 0.002 |
| 2999000 | 93.5 | 92.5 | 80 | 95/95 | 182.504 | 90 | 0.002 |
| 3000000 | 93.5 | 92.3 | 80 | 95/95 | 182.508 | 90 | 0.002 |
