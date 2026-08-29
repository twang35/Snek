# b20j-fc200x50seed2

![b20j-fc200x50seed2 progress](b20j-fc200x50seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.7, perfect games 50%.

## Config

| setting | value |
|---|---|
| policy_name | b20j-fc200x50seed2 |
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

3001 evals so far. Full series in [`b20j-fc200x50seed2_evals.json`](b20j-fc200x50seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.9 | 0.9 | 0 | 2/95 | 0.4 | 0 | 0.4 |
| 2000 | 0.7 | 0.8 | 0 | 3/95 | 0.2 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.6 | 93.88 | 93 | 95/95 | 172.8 | 80 | 0.0032 |
| 2990000 | 93.9 | 93.82 | 92 | 95/95 | 141.35 | 50 | 0.0032 |
| 2991000 | 94.2 | 93.96 | 93 | 95/95 | 151.6 | 60 | 0.0032 |
| 2992000 | 94.3 | 94.18 | 92 | 95/95 | 162.55 | 70 | 0.0032 |
| 2993000 | 93.8 | 94.16 | 88 | 95/95 | 162.5 | 70 | 0.0031 |
| 2994000 | 92.6 | 93.76 | 89 | 95/95 | 110.2 | 20 | 0.0032 |
| 2995000 | 94.6 | 93.9 | 93 | 95/95 | 172.8 | 80 | 0.0031 |
| 2996000 | 93.3 | 93.72 | 86 | 95/95 | 140.75 | 50 | 0.0031 |
| 2997000 | 94.2 | 93.7 | 93 | 95/95 | 152.05 | 60 | 0.003 |
| 2998000 | 93.8 | 93.7 | 90 | 95/95 | 152.1 | 60 | 0.003 |
| 2999000 | 93.8 | 93.94 | 91 | 95/95 | 140.8 | 50 | 0.003 |
| 3000000 | 93.7 | 93.76 | 91 | 95/95 | 141.6 | 50 | 0.0031 |
