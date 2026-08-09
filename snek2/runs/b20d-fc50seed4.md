# b20d-fc50seed4

![b20d-fc50seed4 progress](b20d-fc50seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.6, perfect games 40%.

Training was resumed at step 2734000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | b20d-fc50seed4 |
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
| fc_layer_params | (50, 100, 50) |
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

3001 evals so far. Full series in [`b20d-fc50seed4_evals.json`](b20d-fc50seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 0 | 1/95 | -4.5 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | 0.2 | 0 | 0.4 |
| 2000 | 1.6 | 1.15 | 0 | 6/95 | 1.1 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 92.4 | 93.58 | 86 | 95/95 | 110.0 | 20 | 0.0035 |
| 2990000 | 93.2 | 93.44 | 86 | 95/95 | 141.1 | 50 | 0.0035 |
| 2991000 | 91.9 | 93.1 | 82 | 95/95 | 129.85 | 40 | 0.0036 |
| 2992000 | 93.6 | 92.9 | 91 | 95/95 | 141.5 | 50 | 0.0036 |
| 2993000 | 94.2 | 93.06 | 87 | 95/95 | 182.8 | 90 | 0.0035 |
| 2994000 | 94.1 | 93.4 | 91 | 95/95 | 151.95 | 60 | 0.0035 |
| 2995000 | 94.0 | 93.56 | 91 | 95/95 | 163.15 | 70 | 0.0034 |
| 2996000 | 93.4 | 93.86 | 89 | 95/95 | 141.3 | 50 | 0.0033 |
| 2997000 | 93.7 | 93.88 | 91 | 95/95 | 131.65 | 40 | 0.0034 |
| 2998000 | 93.5 | 93.74 | 91 | 95/95 | 131.9 | 40 | 0.0034 |
| 2999000 | 93.8 | 93.68 | 90 | 95/95 | 141.7 | 50 | 0.0035 |
| 3000000 | 93.6 | 93.6 | 91 | 95/95 | 130.65 | 40 | 0.0035 |
