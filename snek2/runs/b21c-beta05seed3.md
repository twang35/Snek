# b21c-beta05seed3

![b21c-beta05seed3 progress](b21c-beta05seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 89.2, perfect games 60%.

Training was resumed at step 155000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | b21c-beta05seed3 |
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
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | td_error |
| importance_sampling_beta | 0.4 -> 0.5 over 300000 steps |
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

3001 evals so far. Full series in [`b21c-beta05seed3_evals.json`](b21c-beta05seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 1.0 | 1.0 | 0 | 4/95 | 0.5 | 0 | 0.4 |
| 2000 | 1.4 | 1.2 | 0 | 3/95 | 0.9 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 93.8 | 93.58 | 91 | 95/95 | 141.25 | 50 | 0.0041 |
| 2990000 | 92.6 | 93.22 | 85 | 95/95 | 130.1 | 40 | 0.0041 |
| 2991000 | 89.9 | 92.4 | 62 | 95/95 | 106.6 | 20 | 0.0042 |
| 2992000 | 93.5 | 92.78 | 86 | 95/95 | 130.55 | 40 | 0.0042 |
| 2993000 | 94.3 | 92.82 | 91 | 95/95 | 151.7 | 60 | 0.0042 |
| 2994000 | 94.1 | 92.88 | 92 | 95/95 | 141.1 | 50 | 0.0043 |
| 2995000 | 94.0 | 93.16 | 91 | 95/95 | 141.0 | 50 | 0.0043 |
| 2996000 | 94.0 | 93.98 | 91 | 95/95 | 141.0 | 50 | 0.0043 |
| 2997000 | 94.3 | 94.14 | 91 | 95/95 | 162.1 | 70 | 0.0042 |
| 2998000 | 87.0 | 92.68 | 38 | 95/95 | 124.5 | 40 | 0.0042 |
| 2999000 | 91.2 | 92.1 | 64 | 95/95 | 138.65 | 50 | 0.0042 |
| 3000000 | 89.2 | 91.14 | 46 | 95/95 | 147.05 | 60 | 0.0041 |
