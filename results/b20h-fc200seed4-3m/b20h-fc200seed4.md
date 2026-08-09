# b20h-fc200seed4

![b20h-fc200seed4 progress](b20h-fc200seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 94.4, perfect games 60%.

Training was resumed at step 1840000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | b20h-fc200seed4 |
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

2960 evals so far. Full series in [`b20h-fc200seed4_evals.json`](b20h-fc200seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.3 | 0.3 | 0 | 1/95 | -4.7 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 1/95 | -0.1 | 0 | 0.4 |
| 2000 | 0.8 | 0.6 | 0 | 2/95 | 0.3 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.0 | 93.86 | 91 | 95/95 | 151.4 | 60 | 0.0032 |
| 2990000 | 93.7 | 93.84 | 89 | 95/95 | 130.3 | 40 | 0.0032 |
| 2991000 | 93.6 | 93.82 | 90 | 95/95 | 140.6 | 50 | 0.0033 |
| 2992000 | 90.8 | 93.26 | 54 | 95/95 | 169.45 | 80 | 0.0032 |
| 2993000 | 93.7 | 93.16 | 91 | 95/95 | 130.3 | 40 | 0.0034 |
| 2994000 | 94.6 | 93.28 | 93 | 95/95 | 162.4 | 70 | 0.0033 |
| 2995000 | 94.2 | 93.38 | 93 | 95/95 | 151.6 | 60 | 0.0033 |
| 2996000 | 94.8 | 93.62 | 94 | 95/95 | 173.0 | 80 | 0.0033 |
| 2997000 | 93.4 | 94.14 | 91 | 95/95 | 109.2 | 20 | 0.0034 |
| 2998000 | 94.2 | 94.24 | 93 | 95/95 | 151.6 | 60 | 0.0034 |
| 2999000 | 94.1 | 94.14 | 89 | 95/95 | 161.9 | 70 | 0.0034 |
| 3000000 | 94.4 | 94.18 | 93 | 95/95 | 151.8 | 60 | 0.0035 |
