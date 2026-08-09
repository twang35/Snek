# b20b-fc50seed2

![b20b-fc50seed2 progress](b20b-fc50seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2787000, avg score 94.4, perfect games 70%.

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

2788 evals so far. Full series in [`b20b-fc50seed2_evals.json`](b20b-fc50seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.9 | 0.9 | 0 | 3/95 | 0.4 | 0 | 0.4 |
| 2000 | 1.3 | 1.1 | 0 | 4/95 | 0.8 | 0 | 0.4 |
| ... | | | | | | | |
| 2776000 | 93.7 | 93.76 | 86 | 95/95 | 161.95 | 70 | 0.003 |
| 2777000 | 94.0 | 93.82 | 93 | 95/95 | 141.0 | 50 | 0.003 |
| 2778000 | 93.9 | 93.8 | 88 | 95/95 | 162.15 | 70 | 0.003 |
| 2779000 | 94.8 | 94.02 | 93 | 95/95 | 183.4 | 90 | 0.0029 |
| 2780000 | 93.8 | 94.04 | 92 | 95/95 | 141.7 | 50 | 0.0029 |
| 2781000 | 94.2 | 94.14 | 93 | 95/95 | 151.6 | 60 | 0.0029 |
| 2782000 | 94.0 | 94.14 | 93 | 95/95 | 141.0 | 50 | 0.003 |
| 2783000 | 94.0 | 94.16 | 92 | 95/95 | 141.45 | 50 | 0.0029 |
| 2784000 | 94.2 | 94.04 | 93 | 95/95 | 152.05 | 60 | 0.003 |
| 2785000 | 93.0 | 93.88 | 88 | 95/95 | 130.5 | 40 | 0.003 |
| 2786000 | 94.2 | 93.88 | 93 | 95/95 | 151.6 | 60 | 0.0031 |
| 2787000 | 94.4 | 93.96 | 93 | 95/95 | 162.2 | 70 | 0.0031 |
