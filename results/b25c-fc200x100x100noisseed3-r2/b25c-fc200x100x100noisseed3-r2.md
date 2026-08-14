# b25c-fc200x100x100noisseed3-r2

![b25c-fc200x100x100noisseed3-r2 progress](b25c-fc200x100x100noisseed3-r2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 95.0, perfect games 100%.

## Config

| setting | value |
|---|---|
| policy_name | b25c-fc200x100x100noisseed3-r2 |
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

3001 evals so far. Full series in [`b25c-fc200x100x100noisseed3-r2_evals.json`](b25c-fc200x100x100noisseed3-r2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 1.3 | 1.3 | 0 | 4/95 | 0.8 | 0 | 0.4 |
| 2000 | 0.9 | 1.1 | 0 | 4/95 | 0.4 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 93.0 | 90.98 | 84 | 95/95 | 161.7 | 70 | 0.002 |
| 2990000 | 67.6 | 89.12 | 2 | 95/95 | 126.35 | 60 | 0.002 |
| 2991000 | 85.8 | 87.28 | 5 | 95/95 | 164.45 | 80 | 0.002 |
| 2992000 | 95.0 | 87.28 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 2993000 | 95.0 | 87.28 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 2994000 | 94.8 | 87.64 | 93 | 95/95 | 183.4 | 90 | 0.002 |
| 2995000 | 95.0 | 93.12 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 2996000 | 75.8 | 91.12 | 3 | 95/95 | 144.95 | 70 | 0.002 |
| 2997000 | 76.4 | 87.4 | 3 | 95/95 | 145.55 | 70 | 0.002 |
| 2998000 | 94.8 | 87.36 | 93 | 95/95 | 183.4 | 90 | 0.002 |
| 2999000 | 86.0 | 85.6 | 5 | 95/95 | 175.05 | 90 | 0.002 |
| 3000000 | 95.0 | 85.6 | 95 | 95/95 | 194.0 | 100 | 0.002 |
