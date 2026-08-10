# b20i-fc200x50seed1

![b20i-fc200x50seed1 progress](b20i-fc200x50seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.3, perfect games 80%.

## Config

| setting | value |
|---|---|
| policy_name | b20i-fc200x50seed1 |
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
| fc_layer_params | (200, 50) |
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

3001 evals so far. Full series in [`b20i-fc200x50seed1_evals.json`](b20i-fc200x50seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.4 | 0.4 | 0 | 3/95 | -4.15 | 0 | 0.4 |
| 1000 | 1.2 | 1.2 | 0 | 3/95 | 0.7 | 0 | 0.4 |
| 2000 | 2.9 | 2.05 | 0 | 4/95 | 2.4 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.5 | 93.56 | 93 | 95/95 | 162.75 | 70 | 0.0034 |
| 2990000 | 93.5 | 93.86 | 91 | 95/95 | 121.5 | 30 | 0.0035 |
| 2991000 | 91.5 | 93.62 | 82 | 95/95 | 120.85 | 30 | 0.0036 |
| 2992000 | 94.2 | 93.58 | 93 | 95/95 | 153.4 | 60 | 0.0036 |
| 2993000 | 92.1 | 93.16 | 74 | 95/95 | 151.3 | 60 | 0.0036 |
| 2994000 | 92.8 | 92.82 | 82 | 95/95 | 141.6 | 50 | 0.0037 |
| 2995000 | 94.1 | 92.94 | 92 | 95/95 | 152.85 | 60 | 0.0038 |
| 2996000 | 94.3 | 93.5 | 92 | 95/95 | 163.0 | 70 | 0.0037 |
| 2997000 | 93.5 | 93.36 | 86 | 95/95 | 152.25 | 60 | 0.0037 |
| 2998000 | 93.6 | 93.66 | 93 | 95/95 | 121.15 | 30 | 0.0038 |
| 2999000 | 93.5 | 93.8 | 84 | 95/95 | 162.65 | 70 | 0.0038 |
| 3000000 | 93.3 | 93.64 | 82 | 95/95 | 171.95 | 80 | 0.0037 |
