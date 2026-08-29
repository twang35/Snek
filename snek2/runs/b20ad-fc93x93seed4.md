# b20ad-fc93x93seed4

![b20ad-fc93x93seed4 progress](b20ad-fc93x93seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 94.2, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | b20ad-fc93x93seed4 |
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
| fc_layer_params | (93, 93) |
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

3001 evals so far. Full series in [`b20ad-fc93x93seed4_evals.json`](b20ad-fc93x93seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.3 | 0.3 | 0 | 1/95 | -4.7 | 0 | 0.4 |
| 1000 | 4.6 | 4.6 | 0 | 9/95 | 2.3 | 0 | 0.4 |
| 2000 | 0.8 | 2.7 | 0 | 3/95 | 0.3 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.2 | 94.02 | 91 | 95/95 | 162.0 | 70 | 0.0031 |
| 2990000 | 93.1 | 93.88 | 80 | 95/95 | 161.35 | 70 | 0.0031 |
| 2991000 | 94.6 | 93.96 | 93 | 95/95 | 173.25 | 80 | 0.003 |
| 2992000 | 94.5 | 93.9 | 92 | 95/95 | 173.15 | 80 | 0.003 |
| 2993000 | 94.4 | 94.16 | 93 | 95/95 | 162.65 | 70 | 0.0029 |
| 2994000 | 94.4 | 94.2 | 93 | 95/95 | 163.1 | 70 | 0.0029 |
| 2995000 | 93.4 | 94.26 | 91 | 95/95 | 120.5 | 30 | 0.003 |
| 2996000 | 94.3 | 94.2 | 92 | 95/95 | 162.1 | 70 | 0.003 |
| 2997000 | 92.5 | 93.8 | 82 | 95/95 | 129.55 | 40 | 0.003 |
| 2998000 | 93.5 | 93.62 | 86 | 95/95 | 151.35 | 60 | 0.003 |
| 2999000 | 94.4 | 93.62 | 93 | 95/95 | 163.1 | 70 | 0.003 |
| 3000000 | 94.2 | 93.78 | 93 | 95/95 | 151.6 | 60 | 0.003 |
