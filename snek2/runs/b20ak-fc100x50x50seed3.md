# b20ak-fc100x50x50seed3

![b20ak-fc100x50x50seed3 progress](b20ak-fc100x50x50seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.8, perfect games 40%.

## Config

| setting | value |
|---|---|
| policy_name | b20ak-fc100x50x50seed3 |
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

3001 evals so far. Full series in [`b20ak-fc100x50x50seed3_evals.json`](b20ak-fc100x50x50seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.4 | 0.4 | 0 | 2/95 | -4.6 | 0 | 0.4 |
| 1000 | 0.8 | 0.8 | 0 | 7/95 | 0.3 | 0 | 0.4 |
| 2000 | 0.1 | 0.45 | 0 | 1/95 | -0.4 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 93.7 | 93.82 | 91 | 95/95 | 130.3 | 40 | 0.0047 |
| 2990000 | 94.0 | 93.8 | 91 | 95/95 | 151.4 | 60 | 0.0046 |
| 2991000 | 93.5 | 93.68 | 91 | 95/95 | 130.1 | 40 | 0.0046 |
| 2992000 | 93.9 | 93.74 | 91 | 95/95 | 151.3 | 60 | 0.0045 |
| 2993000 | 93.6 | 93.74 | 91 | 95/95 | 140.6 | 50 | 0.0045 |
| 2994000 | 94.5 | 93.9 | 91 | 95/95 | 172.7 | 80 | 0.0044 |
| 2995000 | 94.0 | 93.9 | 91 | 95/95 | 141.0 | 50 | 0.0044 |
| 2996000 | 94.0 | 94.0 | 89 | 95/95 | 151.4 | 60 | 0.0043 |
| 2997000 | 94.7 | 94.16 | 93 | 95/95 | 172.9 | 80 | 0.0042 |
| 2998000 | 94.6 | 94.36 | 92 | 95/95 | 172.8 | 80 | 0.004 |
| 2999000 | 93.9 | 94.24 | 91 | 95/95 | 140.9 | 50 | 0.004 |
| 3000000 | 93.8 | 94.2 | 92 | 95/95 | 130.4 | 40 | 0.0041 |
