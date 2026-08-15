# b30c-chase10fc200x100x100seed3

![b30c-chase10fc200x100x100seed3 progress](b30c-chase10fc200x100x100seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 138000, avg score 92.3, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b30c-chase10fc200x100x100seed3 |
| seed | 3 |
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

139 evals so far. Full series in [`b30c-chase10fc200x100x100seed3_evals.json`](b30c-chase10fc200x100x100seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 1.3 | 1.3 | 0 | 4/95 | 0.8 | 0 | 0.4 |
| 2000 | 0.9 | 1.1 | 0 | 4/95 | 0.4 | 0 | 0.4 |
| ... | | | | | | | |
| 127000 | 93.3 | 91.08 | 90 | 95/95 | 142.084 | 0 | 0.0125 |
| 128000 | 88.8 | 90.42 | 81 | 95/95 | 106.839 | 0 | 0.0125 |
| 129000 | 90.9 | 90.64 | 75 | 95/95 | 120.24 | 0 | 0.0125 |
| 130000 | 87.8 | 90.08 | 77 | 95/95 | 96.787 | 0 | 0.0125 |
| 131000 | 89.9 | 90.14 | 84 | 95/95 | 98.435 | 0 | 0.0125 |
| 132000 | 90.5 | 89.58 | 81 | 95/95 | 128.889 | 0 | 0.0125 |
| 133000 | 90.3 | 89.88 | 79 | 95/95 | 108.783 | 0 | 0.0125 |
| 134000 | 92.1 | 90.12 | 87 | 95/95 | 141.335 | 0 | 0.0125 |
| 135000 | 91.1 | 90.78 | 79 | 95/95 | 129.936 | 0 | 0.0125 |
| 136000 | 90.6 | 90.92 | 82 | 95/95 | 119.034 | 0 | 0.0125 |
| 137000 | 92.4 | 91.3 | 81 | 95/95 | 151.586 | 0 | 0.0125 |
| 138000 | 92.3 | 91.7 | 83 | 95/95 | 130.684 | 0 | 0.0125 |
