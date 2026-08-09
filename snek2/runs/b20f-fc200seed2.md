# b20f-fc200seed2

![b20f-fc200seed2 progress](b20f-fc200seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 92.9, perfect games 40%.

Training was resumed at step 1802000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | b20f-fc200seed2 |
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

2966 evals so far. Full series in [`b20f-fc200seed2_evals.json`](b20f-fc200seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 2/95 | 0.2 | 0 | 0.4 |
| 2000 | 0.6 | 0.65 | 0 | 2/95 | 0.1 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.1 | 93.8 | 90 | 95/95 | 162.35 | 70 | 0.0034 |
| 2990000 | 91.5 | 93.28 | 82 | 95/95 | 128.55 | 40 | 0.0034 |
| 2991000 | 93.8 | 93.12 | 89 | 95/95 | 140.8 | 50 | 0.0034 |
| 2992000 | 92.8 | 93.16 | 88 | 95/95 | 129.85 | 40 | 0.0035 |
| 2993000 | 92.5 | 92.94 | 77 | 95/95 | 129.1 | 40 | 0.0035 |
| 2994000 | 94.7 | 93.06 | 92 | 95/95 | 183.3 | 90 | 0.0033 |
| 2995000 | 92.9 | 93.34 | 81 | 95/95 | 150.75 | 60 | 0.0034 |
| 2996000 | 93.1 | 93.2 | 88 | 95/95 | 119.75 | 30 | 0.0034 |
| 2997000 | 94.0 | 93.44 | 92 | 95/95 | 131.05 | 40 | 0.0035 |
| 2998000 | 91.4 | 93.22 | 65 | 95/95 | 169.6 | 80 | 0.0034 |
| 2999000 | 94.2 | 93.12 | 92 | 95/95 | 141.2 | 50 | 0.0034 |
| 3000000 | 92.9 | 93.12 | 86 | 95/95 | 129.5 | 40 | 0.0034 |
