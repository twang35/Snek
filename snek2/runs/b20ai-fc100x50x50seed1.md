# b20ai-fc100x50x50seed1

![b20ai-fc100x50x50seed1 progress](b20ai-fc100x50x50seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.9, perfect games 50%.

## Config

| setting | value |
|---|---|
| policy_name | b20ai-fc100x50x50seed1 |
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

3001 evals so far. Full series in [`b20ai-fc100x50x50seed1_evals.json`](b20ai-fc100x50x50seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.2 | 0.2 | 0 | 1/95 | -4.8 | 0 | 0.4 |
| 1000 | 1.1 | 1.1 | 0 | 3/95 | 0.6 | 0 | 0.4 |
| 2000 | 2.1 | 1.6 | 0 | 6/95 | 1.6 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 92.9 | 93.92 | 91 | 95/95 | 121.8 | 30 | 0.0037 |
| 2990000 | 94.5 | 94.06 | 92 | 95/95 | 173.15 | 80 | 0.0036 |
| 2991000 | 90.0 | 93.22 | 62 | 95/95 | 118.0 | 30 | 0.0037 |
| 2992000 | 94.4 | 93.18 | 93 | 95/95 | 163.1 | 70 | 0.0036 |
| 2993000 | 94.4 | 93.24 | 90 | 95/95 | 173.05 | 80 | 0.0035 |
| 2994000 | 93.4 | 93.34 | 91 | 95/95 | 110.1 | 20 | 0.0035 |
| 2995000 | 94.4 | 93.32 | 93 | 95/95 | 162.65 | 70 | 0.0035 |
| 2996000 | 94.0 | 94.12 | 92 | 95/95 | 151.85 | 60 | 0.0035 |
| 2997000 | 93.3 | 93.9 | 90 | 95/95 | 142.55 | 50 | 0.0035 |
| 2998000 | 92.8 | 93.58 | 80 | 95/95 | 151.1 | 60 | 0.0035 |
| 2999000 | 94.6 | 93.82 | 93 | 95/95 | 173.7 | 80 | 0.0035 |
| 3000000 | 93.9 | 93.72 | 92 | 95/95 | 141.8 | 50 | 0.0035 |
