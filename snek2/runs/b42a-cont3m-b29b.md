# b42a-cont3m-b29b

![b42a-cont3m-b29b progress](b42a-cont3m-b29b.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1832000, avg score 95.0, perfect games 100%.

## Config

| setting | value |
|---|---|
| policy_name | b42a-cont3m-b29b |
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

386 evals so far. Full series in [`b42a-cont3m-b29b_evals.json`](b42a-cont3m-b29b_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1447000 | 93.8 | 93.8 | 86 | 95/95 | 172.859 | 80 | 0.002 |
| 1448000 | 95.0 | 95.0 | 86 | 95/95 | 193.952 | 100 | 0.002 |
| 1449000 | 95.0 | 95.0 | 95 | 95/95 | 193.951 | 100 | 0.002 |
| ... | | | | | | | |
| 1821000 | 85.2 | 89.36 | 0 | 95/95 | 164.256 | 80 | 0.002 |
| 1822000 | 94.6 | 89.28 | 91 | 95/95 | 183.607 | 90 | 0.002 |
| 1823000 | 91.8 | 88.64 | 66 | 95/95 | 170.854 | 80 | 0.002 |
| 1824000 | 95.0 | 90.52 | 95 | 95/95 | 193.958 | 100 | 0.002 |
| 1825000 | 95.0 | 92.32 | 95 | 95/95 | 193.959 | 100 | 0.002 |
| 1826000 | 88.6 | 93.0 | 36 | 95/95 | 157.711 | 70 | 0.002 |
| 1827000 | 86.0 | 91.28 | 5 | 95/95 | 175.005 | 90 | 0.002 |
| 1828000 | 85.6 | 90.04 | 1 | 95/95 | 174.61 | 90 | 0.002 |
| 1829000 | 93.7 | 89.78 | 82 | 95/95 | 182.703 | 90 | 0.002 |
| 1830000 | 91.0 | 88.98 | 66 | 95/95 | 140.201 | 50 | 0.002 |
| 1831000 | 95.0 | 90.26 | 95 | 95/95 | 193.949 | 100 | 0.002 |
| 1832000 | 95.0 | 92.06 | 95 | 95/95 | 193.952 | 100 | 0.002 |
