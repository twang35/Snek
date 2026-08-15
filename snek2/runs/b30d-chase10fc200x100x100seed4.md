# b30d-chase10fc200x100x100seed4

![b30d-chase10fc200x100x100seed4 progress](b30d-chase10fc200x100x100seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 139000, avg score 82.1, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b30d-chase10fc200x100x100seed4 |
| seed | 4 |
| zeroed_observations | none |
| learning_rate | 1e-05 |
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
| fc_layer_params | (200, 100, 100) |
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
| CHASE_SAFE_SHAPING | c=0.1, potential-based on head/food/tail in one region, gated to snake length >= 85 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

140 evals so far. Full series in [`b30d-chase10fc200x100x100seed4_evals.json`](b30d-chase10fc200x100x100seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.6 | 0.6 | 0 | 3/95 | 0.1 | 0 | 0.4 |
| 2000 | 2.0 | 1.3 | 1 | 4/95 | 1.5 | 0 | 0.4 |
| ... | | | | | | | |
| 128000 | 88.3 | 89.12 | 82 | 95/95 | 97.733 | 0 | 0.0125 |
| 129000 | 82.7 | 88.4 | 0 | 95/95 | 102.083 | 0 | 0.0125 |
| 130000 | 87.5 | 87.9 | 82 | 93/95 | 86.985 | 0 | 0.0125 |
| 131000 | 89.2 | 87.56 | 86 | 93/95 | 88.681 | 0 | 0.0125 |
| 132000 | 90.9 | 87.72 | 84 | 95/95 | 110.281 | 0 | 0.0125 |
| 133000 | 90.8 | 88.22 | 86 | 95/95 | 110.181 | 0 | 0.0125 |
| 134000 | 91.3 | 89.94 | 84 | 95/95 | 120.633 | 0 | 0.0125 |
| 135000 | 90.5 | 90.54 | 84 | 95/95 | 119.833 | 0 | 0.0125 |
| 136000 | 92.5 | 91.2 | 90 | 95/95 | 131.781 | 0 | 0.0125 |
| 137000 | 92.0 | 91.42 | 88 | 95/95 | 121.332 | 0 | 0.0125 |
| 138000 | 81.8 | 89.62 | 0 | 95/95 | 110.687 | 0 | 0.0125 |
| 139000 | 82.1 | 87.78 | 1 | 95/95 | 101.482 | 0 | 0.0125 |
