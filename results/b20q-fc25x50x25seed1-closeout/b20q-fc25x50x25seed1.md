# b20q-fc25x50x25seed1

![b20q-fc25x50x25seed1 progress](b20q-fc25x50x25seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.4, perfect games 20%.

## Config

| setting | value |
|---|---|
| policy_name | b20q-fc25x50x25seed1 |
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
| fc_layer_params | (25, 50, 25) |
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

3001 evals so far. Full series in [`b20q-fc25x50x25seed1_evals.json`](b20q-fc25x50x25seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 1/95 | -5.0 | 0 | 0.4 |
| 2000 | 0.2 | 0.1 | 0 | 1/95 | -4.8 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 89.2 | 89.8 | 80 | 95/95 | 95.95 | 10 | 0.0083 |
| 2990000 | 91.9 | 89.8 | 82 | 94/95 | 87.35 | 0 | 0.0083 |
| 2991000 | 89.6 | 90.34 | 76 | 95/95 | 106.3 | 20 | 0.0084 |
| 2992000 | 90.3 | 90.66 | 76 | 95/95 | 106.55 | 20 | 0.0083 |
| 2993000 | 88.1 | 89.82 | 74 | 94/95 | 84.45 | 0 | 0.0087 |
| 2994000 | 92.2 | 90.42 | 83 | 95/95 | 108.45 | 20 | 0.0087 |
| 2995000 | 93.1 | 90.66 | 90 | 95/95 | 119.3 | 30 | 0.0086 |
| 2996000 | 89.6 | 90.66 | 80 | 94/95 | 85.5 | 0 | 0.0087 |
| 2997000 | 92.7 | 91.14 | 90 | 95/95 | 98.1 | 10 | 0.0087 |
| 2998000 | 91.7 | 91.86 | 87 | 95/95 | 107.95 | 20 | 0.0087 |
| 2999000 | 90.3 | 91.48 | 77 | 94/95 | 86.2 | 0 | 0.0089 |
| 3000000 | 93.4 | 91.54 | 90 | 95/95 | 109.65 | 20 | 0.0089 |
