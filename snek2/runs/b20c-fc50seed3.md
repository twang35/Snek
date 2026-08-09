# b20c-fc50seed3

![b20c-fc50seed3 progress](b20c-fc50seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2561000, avg score 94.6, perfect games 80%.

## Config

| setting | value |
|---|---|
| policy_name | b20c-fc50seed3 |
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
| importance_sampling_beta | 0.4 -> 1.0 over 300000 steps |
| max_steps | 10000000 |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.0 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2562 evals so far. Full series in [`b20c-fc50seed3_evals.json`](b20c-fc50seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 0.9 | 0.9 | 0 | 4/95 | 0.4 | 0 | 0.4 |
| 2000 | 1.2 | 1.05 | 0 | 3/95 | 0.7 | 0 | 0.4 |
| ... | | | | | | | |
| 2550000 | 93.8 | 93.58 | 93 | 95/95 | 130.4 | 40 | 0.0048 |
| 2551000 | 93.6 | 93.72 | 89 | 95/95 | 141.05 | 50 | 0.0048 |
| 2552000 | 94.0 | 93.82 | 92 | 95/95 | 141.45 | 50 | 0.0047 |
| 2553000 | 94.3 | 93.8 | 93 | 95/95 | 152.15 | 60 | 0.0046 |
| 2554000 | 93.9 | 93.92 | 93 | 95/95 | 130.95 | 40 | 0.0047 |
| 2555000 | 92.6 | 93.68 | 89 | 95/95 | 98.45 | 10 | 0.0048 |
| 2556000 | 93.5 | 93.66 | 91 | 95/95 | 119.7 | 30 | 0.0048 |
| 2557000 | 94.1 | 93.68 | 93 | 95/95 | 141.1 | 50 | 0.0048 |
| 2558000 | 93.7 | 93.56 | 91 | 95/95 | 141.15 | 50 | 0.0047 |
| 2559000 | 93.7 | 93.52 | 93 | 95/95 | 119.9 | 30 | 0.0047 |
| 2560000 | 90.1 | 93.02 | 60 | 95/95 | 116.75 | 30 | 0.0047 |
| 2561000 | 94.6 | 93.24 | 93 | 95/95 | 172.8 | 80 | 0.0046 |
