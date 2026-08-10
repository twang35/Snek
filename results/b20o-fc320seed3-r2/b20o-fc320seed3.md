# b20o-fc320seed3

![b20o-fc320seed3 progress](b20o-fc320seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.9, perfect games 60%.

Training was resumed at step 243000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | b20o-fc320seed3 |
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

3001 evals so far. Full series in [`b20o-fc320seed3_evals.json`](b20o-fc320seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 1.4 | 1.4 | 0 | 3/95 | 0.9 | 0 | 0.4 |
| 2000 | 1.4 | 1.4 | 0 | 7/95 | 0.9 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 93.0 | 92.16 | 84 | 95/95 | 151.3 | 60 | 0.0039 |
| 2990000 | 93.9 | 92.88 | 91 | 95/95 | 142.25 | 50 | 0.0038 |
| 2991000 | 93.8 | 93.06 | 88 | 95/95 | 152.1 | 60 | 0.0038 |
| 2992000 | 92.3 | 93.06 | 80 | 95/95 | 140.65 | 50 | 0.0038 |
| 2993000 | 92.1 | 93.02 | 82 | 95/95 | 131.4 | 40 | 0.0038 |
| 2994000 | 91.8 | 92.78 | 84 | 95/95 | 140.6 | 50 | 0.0039 |
| 2995000 | 93.7 | 92.74 | 91 | 95/95 | 131.65 | 40 | 0.004 |
| 2996000 | 94.0 | 92.78 | 91 | 95/95 | 152.75 | 60 | 0.0039 |
| 2997000 | 92.7 | 92.86 | 86 | 95/95 | 131.55 | 40 | 0.004 |
| 2998000 | 91.5 | 92.74 | 82 | 95/95 | 139.85 | 50 | 0.004 |
| 2999000 | 92.6 | 92.9 | 82 | 95/95 | 130.55 | 40 | 0.0041 |
| 3000000 | 93.9 | 92.94 | 90 | 95/95 | 152.65 | 60 | 0.0041 |
