# b30a-chase10fc200x100x100seed1

![b30a-chase10fc200x100x100seed1 progress](b30a-chase10fc200x100x100seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 138000, avg score 93.4, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b30a-chase10fc200x100x100seed1 |
| seed | 1 |
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

139 evals so far. Full series in [`b30a-chase10fc200x100x100seed1_evals.json`](b30a-chase10fc200x100x100seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 1.4 | 1.4 | 0 | 3/95 | 0.9 | 0 | 0.4 |
| 2000 | 1.4 | 1.4 | 0 | 5/95 | 0.9 | 0 | 0.4 |
| ... | | | | | | | |
| 127000 | 86.0 | 90.86 | 26 | 95/95 | 125.287 | 0 | 0.0125 |
| 128000 | 93.9 | 91.12 | 92 | 95/95 | 153.082 | 0 | 0.0125 |
| 129000 | 92.4 | 91.2 | 74 | 95/95 | 161.535 | 0 | 0.0125 |
| 130000 | 93.5 | 91.26 | 88 | 95/95 | 162.633 | 0 | 0.0125 |
| 131000 | 90.9 | 91.34 | 72 | 95/95 | 130.185 | 0 | 0.0125 |
| 132000 | 94.5 | 93.04 | 92 | 95/95 | 173.583 | 0 | 0.0125 |
| 133000 | 88.9 | 92.04 | 81 | 95/95 | 98.335 | 0 | 0.0125 |
| 134000 | 92.7 | 92.1 | 88 | 95/95 | 131.981 | 0 | 0.0125 |
| 135000 | 91.2 | 91.64 | 78 | 95/95 | 130.484 | 0 | 0.0125 |
| 136000 | 92.0 | 91.86 | 88 | 95/95 | 111.383 | 0 | 0.0125 |
| 137000 | 93.5 | 91.66 | 88 | 95/95 | 152.685 | 0 | 0.0125 |
| 138000 | 93.4 | 92.56 | 89 | 95/95 | 152.584 | 0 | 0.0125 |
