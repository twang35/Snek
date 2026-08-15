# b25d-fc200x100x100noisseed4-r2

![b25d-fc200x100x100noisseed4-r2 progress](b25d-fc200x100x100noisseed4-r2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 94.5, perfect games 80%.

## Config

| setting | value |
|---|---|
| policy_name | b25d-fc200x100x100noisseed4-r2 |
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

3001 evals so far. Full series in [`b25d-fc200x100x100noisseed4-r2_evals.json`](b25d-fc200x100x100noisseed4-r2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.6 | 0.6 | 0 | 3/95 | 0.1 | 0 | 0.4 |
| 2000 | 1.4 | 1.0 | 0 | 4/95 | 0.9 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.9 | 92.92 | 94 | 95/95 | 183.5 | 90 | 0.002 |
| 2990000 | 95.0 | 92.98 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 2991000 | 94.7 | 93.06 | 93 | 95/95 | 172.9 | 80 | 0.002 |
| 2992000 | 93.2 | 94.56 | 83 | 95/95 | 161.9 | 70 | 0.002 |
| 2993000 | 92.7 | 94.1 | 86 | 95/95 | 130.2 | 40 | 0.002 |
| 2994000 | 95.0 | 94.12 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 2995000 | 94.6 | 94.04 | 91 | 95/95 | 183.65 | 90 | 0.002 |
| 2996000 | 94.6 | 94.02 | 92 | 95/95 | 172.8 | 80 | 0.002 |
| 2997000 | 94.3 | 94.24 | 92 | 95/95 | 162.1 | 70 | 0.002 |
| 2998000 | 94.9 | 94.68 | 94 | 95/95 | 183.5 | 90 | 0.002 |
| 2999000 | 95.0 | 94.68 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 3000000 | 94.5 | 94.66 | 92 | 95/95 | 172.7 | 80 | 0.002 |
