# b20e-fc200seed1

![b20e-fc200seed1 progress](b20e-fc200seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.2, perfect games 20%.

Training was resumed at step 1790000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | b20e-fc200seed1 |
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

2964 evals so far. Full series in [`b20e-fc200seed1_evals.json`](b20e-fc200seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.0 | 0 | 0.4 |
| 1000 | 1.3 | 1.3 | 0 | 8/95 | 0.8 | 0 | 0.4 |
| 2000 | 1.2 | 1.25 | 0 | 4/95 | 0.7 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 93.6 | 93.52 | 91 | 95/95 | 140.6 | 50 | 0.0034 |
| 2990000 | 91.9 | 93.1 | 86 | 95/95 | 107.7 | 20 | 0.0035 |
| 2991000 | 94.0 | 93.7 | 91 | 95/95 | 151.4 | 60 | 0.0035 |
| 2992000 | 93.1 | 93.4 | 91 | 95/95 | 98.5 | 10 | 0.0036 |
| 2993000 | 92.4 | 93.0 | 83 | 95/95 | 129.45 | 40 | 0.0036 |
| 2994000 | 92.9 | 92.86 | 87 | 95/95 | 119.1 | 30 | 0.0036 |
| 2995000 | 93.8 | 93.24 | 91 | 95/95 | 151.2 | 60 | 0.0036 |
| 2996000 | 94.0 | 93.24 | 89 | 95/95 | 161.8 | 70 | 0.0036 |
| 2997000 | 92.2 | 93.06 | 79 | 95/95 | 118.85 | 30 | 0.0037 |
| 2998000 | 93.1 | 93.2 | 91 | 95/95 | 129.7 | 40 | 0.0038 |
| 2999000 | 91.6 | 92.94 | 73 | 95/95 | 139.05 | 50 | 0.0038 |
| 3000000 | 93.2 | 92.82 | 91 | 95/95 | 109.0 | 20 | 0.0039 |
