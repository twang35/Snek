# b20p-fc320seed4

![b20p-fc320seed4 progress](b20p-fc320seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 94.4, perfect games 80%.

Training was resumed at step 246000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | b20p-fc320seed4 |
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

3001 evals so far. Full series in [`b20p-fc320seed4_evals.json`](b20p-fc320seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 18.5 | 18.5 | 0 | 73/95 | 17.1 | 0 | 0.1 |
| 2000 | 55.9 | 37.2 | 28 | 77/95 | 53.15 | 0 | 0.0125 |
| ... | | | | | | | |
| 2989000 | 94.5 | 94.28 | 90 | 95/95 | 183.55 | 90 | 0.0024 |
| 2990000 | 89.8 | 93.52 | 52 | 95/95 | 147.65 | 60 | 0.0025 |
| 2991000 | 94.4 | 93.44 | 93 | 95/95 | 162.2 | 70 | 0.0025 |
| 2992000 | 95.0 | 93.56 | 95 | 95/95 | 194.0 | 100 | 0.0024 |
| 2993000 | 94.2 | 93.58 | 92 | 95/95 | 162.9 | 70 | 0.0024 |
| 2994000 | 93.6 | 93.4 | 87 | 95/95 | 161.85 | 70 | 0.0024 |
| 2995000 | 91.5 | 93.74 | 60 | 95/95 | 180.1 | 90 | 0.0023 |
| 2996000 | 94.5 | 93.76 | 90 | 95/95 | 183.55 | 90 | 0.0023 |
| 2997000 | 93.9 | 93.54 | 91 | 95/95 | 151.75 | 60 | 0.0023 |
| 2998000 | 95.0 | 93.7 | 95 | 95/95 | 194.0 | 100 | 0.0023 |
| 2999000 | 94.4 | 93.86 | 93 | 95/95 | 162.2 | 70 | 0.0023 |
| 3000000 | 94.4 | 94.44 | 92 | 95/95 | 173.5 | 80 | 0.0022 |
