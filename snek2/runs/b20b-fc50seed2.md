# b20b-fc50seed2

![b20b-fc50seed2 progress](b20b-fc50seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 94.4, perfect games 70%.

Training was resumed at step 2787000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | b20b-fc50seed2 |
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

3001 evals so far. Full series in [`b20b-fc50seed2_evals.json`](b20b-fc50seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.9 | 0.9 | 0 | 3/95 | 0.4 | 0 | 0.4 |
| 2000 | 1.3 | 1.1 | 0 | 4/95 | 0.8 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.6 | 92.84 | 93 | 95/95 | 172.8 | 80 | 0.003 |
| 2990000 | 92.9 | 92.54 | 88 | 95/95 | 120.0 | 30 | 0.003 |
| 2991000 | 92.5 | 92.28 | 86 | 95/95 | 130.0 | 40 | 0.0031 |
| 2992000 | 94.0 | 92.26 | 93 | 95/95 | 141.0 | 50 | 0.0031 |
| 2993000 | 93.4 | 93.48 | 88 | 95/95 | 130.45 | 40 | 0.0033 |
| 2994000 | 94.4 | 93.44 | 93 | 95/95 | 162.2 | 70 | 0.0033 |
| 2995000 | 93.9 | 93.64 | 88 | 95/95 | 162.15 | 70 | 0.0033 |
| 2996000 | 94.1 | 93.96 | 92 | 95/95 | 151.95 | 60 | 0.0033 |
| 2997000 | 93.7 | 93.9 | 88 | 95/95 | 151.55 | 60 | 0.0033 |
| 2998000 | 94.2 | 94.06 | 93 | 95/95 | 151.6 | 60 | 0.0032 |
| 2999000 | 93.8 | 93.94 | 88 | 95/95 | 172.9 | 80 | 0.0031 |
| 3000000 | 94.4 | 94.04 | 93 | 95/95 | 162.2 | 70 | 0.0031 |
