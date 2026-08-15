# b25a-fc200x100x100noisseed1-r2

![b25a-fc200x100x100noisseed1-r2 progress](b25a-fc200x100x100noisseed1-r2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.4, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | b25a-fc200x100x100noisseed1-r2 |
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

3001 evals so far. Full series in [`b25a-fc200x100x100noisseed1-r2_evals.json`](b25a-fc200x100x100noisseed1-r2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 1.6 | 1.6 | 0 | 4/95 | 1.1 | 0 | 0.4 |
| 2000 | 1.7 | 1.65 | 0 | 6/95 | 1.2 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 93.8 | 93.12 | 83 | 95/95 | 182.85 | 90 | 0.0022 |
| 2990000 | 94.4 | 93.88 | 91 | 95/95 | 173.05 | 80 | 0.0022 |
| 2991000 | 94.4 | 94.0 | 91 | 95/95 | 173.05 | 80 | 0.0022 |
| 2992000 | 95.0 | 94.36 | 95 | 95/95 | 194.0 | 100 | 0.0021 |
| 2993000 | 94.8 | 94.48 | 93 | 95/95 | 183.4 | 90 | 0.0021 |
| 2994000 | 94.6 | 94.64 | 93 | 95/95 | 172.8 | 80 | 0.0021 |
| 2995000 | 93.5 | 94.46 | 82 | 95/95 | 172.15 | 80 | 0.0021 |
| 2996000 | 94.8 | 94.54 | 93 | 95/95 | 183.4 | 90 | 0.0021 |
| 2997000 | 94.7 | 94.48 | 93 | 95/95 | 172.9 | 80 | 0.0021 |
| 2998000 | 89.9 | 93.5 | 56 | 95/95 | 147.3 | 60 | 0.0022 |
| 2999000 | 94.2 | 93.42 | 91 | 95/95 | 172.4 | 80 | 0.0022 |
| 3000000 | 93.4 | 93.4 | 85 | 95/95 | 151.25 | 60 | 0.0022 |
