# b22a-noisseed1

![b22a-noisseed1 progress](b22a-noisseed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 84.0, perfect games 40%.

## Config

| setting | value |
|---|---|
| policy_name | b22a-noisseed1 |
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
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | td_error |
| importance_sampling_beta | disabled |
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

3001 evals so far. Full series in [`b22a-noisseed1_evals.json`](b22a-noisseed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -0.5 | 0 | 0.4 |
| 2000 | 0.1 | 0.05 | 0 | 1/95 | -0.4 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 90.9 | 89.36 | 80 | 95/95 | 119.8 | 30 | 0.0029 |
| 2990000 | 93.7 | 90.0 | 88 | 95/95 | 152.45 | 60 | 0.0029 |
| 2991000 | 91.8 | 89.96 | 82 | 95/95 | 130.2 | 40 | 0.0029 |
| 2992000 | 85.6 | 90.34 | 12 | 95/95 | 144.8 | 60 | 0.0029 |
| 2993000 | 94.2 | 91.24 | 93 | 95/95 | 152.05 | 60 | 0.0029 |
| 2994000 | 92.6 | 91.58 | 76 | 95/95 | 161.75 | 70 | 0.0029 |
| 2995000 | 90.1 | 90.86 | 78 | 95/95 | 128.95 | 40 | 0.0029 |
| 2996000 | 92.2 | 90.94 | 74 | 95/95 | 150.95 | 60 | 0.0029 |
| 2997000 | 91.0 | 92.02 | 84 | 95/95 | 129.85 | 40 | 0.003 |
| 2998000 | 93.7 | 91.92 | 86 | 95/95 | 162.4 | 70 | 0.0029 |
| 2999000 | 92.5 | 91.9 | 86 | 95/95 | 151.7 | 60 | 0.0029 |
| 3000000 | 84.0 | 90.68 | 11 | 95/95 | 121.5 | 40 | 0.003 |
