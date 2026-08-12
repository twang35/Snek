# b20al-fc100x50x50seed4

![b20al-fc100x50x50seed4 progress](b20al-fc100x50x50seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 94.5, perfect games 80%.

## Config

| setting | value |
|---|---|
| policy_name | b20al-fc100x50x50seed4 |
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
| fc_layer_params | (100, 50, 50) |
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

3001 evals so far. Full series in [`b20al-fc100x50x50seed4_evals.json`](b20al-fc100x50x50seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.3 | 0.3 | 0 | 2/95 | -0.2 | 0 | 0.4 |
| 2000 | 0.7 | 0.5 | 0 | 3/95 | 0.2 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 95.0 | 93.96 | 95 | 95/95 | 194.0 | 100 | 0.0037 |
| 2990000 | 93.1 | 93.9 | 88 | 95/95 | 151.85 | 60 | 0.0037 |
| 2991000 | 93.6 | 93.98 | 89 | 95/95 | 141.95 | 50 | 0.0037 |
| 2992000 | 93.7 | 93.92 | 93 | 95/95 | 122.15 | 30 | 0.0038 |
| 2993000 | 93.2 | 93.72 | 91 | 95/95 | 130.7 | 40 | 0.0037 |
| 2994000 | 93.8 | 93.48 | 91 | 95/95 | 141.25 | 50 | 0.0037 |
| 2995000 | 93.6 | 93.58 | 91 | 95/95 | 141.95 | 50 | 0.0037 |
| 2996000 | 94.2 | 93.7 | 91 | 95/95 | 162.45 | 70 | 0.0037 |
| 2997000 | 93.4 | 93.64 | 89 | 95/95 | 130.45 | 40 | 0.0037 |
| 2998000 | 93.0 | 93.6 | 91 | 95/95 | 120.1 | 30 | 0.0038 |
| 2999000 | 93.1 | 93.46 | 90 | 95/95 | 141.45 | 50 | 0.0038 |
| 3000000 | 94.5 | 93.64 | 92 | 95/95 | 173.15 | 80 | 0.0037 |
