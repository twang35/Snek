# b20g-fc200seed3

![b20g-fc200seed3 progress](b20g-fc200seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 94.8, perfect games 90%.

Training was resumed at step 1772000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | b20g-fc200seed3 |
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
| fc_layer_params | (200, 100, 50) |
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

2963 evals so far. Full series in [`b20g-fc200seed3_evals.json`](b20g-fc200seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.8 | 0.8 | 0 | 4/95 | -4.2 | 0 | 0.4 |
| 1000 | 0.8 | 0.8 | 0 | 4/95 | 0.3 | 0 | 0.4 |
| 2000 | 1.0 | 0.9 | 0 | 2/95 | 0.5 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.0 | 94.62 | 91 | 95/95 | 162.25 | 70 | 0.0021 |
| 2990000 | 94.4 | 94.52 | 91 | 95/95 | 172.6 | 80 | 0.0021 |
| 2991000 | 93.2 | 94.24 | 87 | 95/95 | 130.25 | 40 | 0.0022 |
| 2992000 | 94.3 | 94.14 | 91 | 95/95 | 162.1 | 70 | 0.0022 |
| 2993000 | 94.8 | 94.14 | 93 | 95/95 | 183.4 | 90 | 0.0022 |
| 2994000 | 93.6 | 94.06 | 91 | 95/95 | 130.65 | 40 | 0.0023 |
| 2995000 | 94.5 | 94.08 | 90 | 95/95 | 183.55 | 90 | 0.0023 |
| 2996000 | 94.4 | 94.32 | 91 | 95/95 | 172.6 | 80 | 0.0023 |
| 2997000 | 94.6 | 94.38 | 91 | 95/95 | 183.65 | 90 | 0.0022 |
| 2998000 | 94.8 | 94.38 | 93 | 95/95 | 183.4 | 90 | 0.0022 |
| 2999000 | 94.6 | 94.58 | 93 | 95/95 | 172.8 | 80 | 0.0022 |
| 3000000 | 94.8 | 94.64 | 93 | 95/95 | 183.4 | 90 | 0.0022 |
