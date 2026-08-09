# b20i-fc200x50seed1

![b20i-fc200x50seed1 progress](b20i-fc200x50seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 710000, avg score 87.7, perfect games 10%.

## Config

| setting | value |
|---|---|
| policy_name | b20i-fc200x50seed1 |
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
| fc_layer_params | (200, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | td_error |
| importance_sampling_beta | 0.4 -> 1.0 over 300000 steps |
| max_steps | 3000000 |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.0 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

711 evals so far. Full series in [`b20i-fc200x50seed1_evals.json`](b20i-fc200x50seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.4 | 0.4 | 0 | 3/95 | -4.15 | 0 | 0.4 |
| 1000 | 1.2 | 1.2 | 0 | 3/95 | 0.7 | 0 | 0.4 |
| 2000 | 2.9 | 2.05 | 0 | 4/95 | 2.4 | 0 | 0.4 |
| ... | | | | | | | |
| 699000 | 89.8 | 90.36 | 82 | 93/95 | 89.3 | 0 | 0.0087 |
| 700000 | 87.4 | 89.92 | 80 | 95/95 | 96.85 | 10 | 0.0088 |
| 701000 | 88.1 | 89.26 | 84 | 92/95 | 87.6 | 0 | 0.0088 |
| 702000 | 90.2 | 89.24 | 82 | 93/95 | 89.7 | 0 | 0.0089 |
| 703000 | 87.3 | 88.56 | 80 | 93/95 | 86.8 | 0 | 0.0091 |
| 704000 | 85.9 | 87.78 | 62 | 95/95 | 94.9 | 10 | 0.0093 |
| 705000 | 89.9 | 88.28 | 80 | 95/95 | 119.25 | 30 | 0.0091 |
| 706000 | 89.4 | 88.54 | 80 | 95/95 | 98.85 | 10 | 0.0093 |
| 707000 | 89.5 | 88.4 | 80 | 95/95 | 98.5 | 10 | 0.0094 |
| 708000 | 89.7 | 88.88 | 84 | 95/95 | 99.15 | 10 | 0.0094 |
| 709000 | 91.4 | 89.98 | 84 | 95/95 | 110.8 | 20 | 0.0094 |
| 710000 | 87.7 | 89.54 | 72 | 95/95 | 97.15 | 10 | 0.0096 |
