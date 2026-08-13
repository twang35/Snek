# b24c-fc320noisseed3

![b24c-fc320noisseed3 progress](b24c-fc320noisseed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 94.8, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b24c-fc320noisseed3 |
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
| fc_layer_params | (320,) |
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

3001 evals so far. Full series in [`b24c-fc320noisseed3_evals.json`](b24c-fc320noisseed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 13.3 | 13.3 | 0 | 67/95 | 11.9 | 0 | 0.2 |
| 2000 | 2.1 | 7.7 | 1 | 4/95 | 1.6 | 0 | 0.2 |
| ... | | | | | | | |
| 2989000 | 95.0 | 94.14 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 2990000 | 94.7 | 94.72 | 92 | 95/95 | 183.75 | 90 | 0.002 |
| 2991000 | 95.0 | 94.94 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 2992000 | 95.0 | 94.94 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 2993000 | 94.5 | 94.84 | 92 | 95/95 | 173.15 | 80 | 0.002 |
| 2994000 | 90.4 | 93.92 | 49 | 95/95 | 179.45 | 90 | 0.002 |
| 2995000 | 95.0 | 93.98 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 2996000 | 95.0 | 93.98 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 2997000 | 95.0 | 93.98 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 2998000 | 95.0 | 94.08 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 2999000 | 95.0 | 95.0 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 3000000 | 94.8 | 94.96 | 93 | 95/95 | 183.85 | 90 | 0.002 |
