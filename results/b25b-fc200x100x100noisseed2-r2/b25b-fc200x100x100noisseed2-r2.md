# b25b-fc200x100x100noisseed2-r2

![b25b-fc200x100x100noisseed2-r2 progress](b25b-fc200x100x100noisseed2-r2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 94.1, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | b25b-fc200x100x100noisseed2-r2 |
| seed | 2 |
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

3001 evals so far. Full series in [`b25b-fc200x100x100noisseed2-r2_evals.json`](b25b-fc200x100x100noisseed2-r2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.9 | 0.9 | 0 | 3/95 | 0.4 | 0 | 0.4 |
| 2000 | 1.0 | 0.95 | 0 | 3/95 | 0.5 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 91.3 | 93.56 | 69 | 95/95 | 160.0 | 70 | 0.002 |
| 2990000 | 94.8 | 93.6 | 93 | 95/95 | 183.4 | 90 | 0.002 |
| 2991000 | 94.6 | 93.56 | 93 | 95/95 | 172.8 | 80 | 0.002 |
| 2992000 | 94.0 | 93.44 | 91 | 95/95 | 151.4 | 60 | 0.002 |
| 2993000 | 92.0 | 93.34 | 65 | 95/95 | 181.05 | 90 | 0.002 |
| 2994000 | 94.8 | 94.04 | 93 | 95/95 | 183.4 | 90 | 0.002 |
| 2995000 | 94.2 | 93.92 | 91 | 95/95 | 162.0 | 70 | 0.002 |
| 2996000 | 93.4 | 93.68 | 86 | 95/95 | 172.05 | 80 | 0.002 |
| 2997000 | 94.3 | 93.74 | 92 | 95/95 | 162.1 | 70 | 0.002 |
| 2998000 | 95.0 | 94.34 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 2999000 | 94.8 | 94.34 | 93 | 95/95 | 183.4 | 90 | 0.002 |
| 3000000 | 94.1 | 94.32 | 92 | 95/95 | 151.5 | 60 | 0.002 |
