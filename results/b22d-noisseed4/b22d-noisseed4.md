# b22d-noisseed4

![b22d-noisseed4 progress](b22d-noisseed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.5, perfect games 70%.

## Config

| setting | value |
|---|---|
| policy_name | b22d-noisseed4 |
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

3001 evals so far. Full series in [`b22d-noisseed4_evals.json`](b22d-noisseed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 0 | 1/95 | -4.5 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | 0.2 | 0 | 0.4 |
| 2000 | 0.6 | 0.65 | 0 | 3/95 | 0.1 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.3 | 92.48 | 88 | 95/95 | 183.35 | 90 | 0.0029 |
| 2990000 | 94.1 | 93.18 | 91 | 95/95 | 151.5 | 60 | 0.0029 |
| 2991000 | 93.9 | 94.28 | 87 | 95/95 | 161.7 | 70 | 0.003 |
| 2992000 | 94.3 | 94.22 | 93 | 95/95 | 151.7 | 60 | 0.003 |
| 2993000 | 94.4 | 94.2 | 93 | 95/95 | 152.25 | 60 | 0.0029 |
| 2994000 | 93.1 | 93.96 | 90 | 95/95 | 130.6 | 40 | 0.003 |
| 2995000 | 91.0 | 93.34 | 78 | 95/95 | 109.05 | 20 | 0.003 |
| 2996000 | 94.1 | 93.38 | 90 | 95/95 | 162.35 | 70 | 0.003 |
| 2997000 | 94.5 | 93.42 | 93 | 95/95 | 162.3 | 70 | 0.0029 |
| 2998000 | 86.9 | 91.92 | 28 | 95/95 | 134.8 | 50 | 0.003 |
| 2999000 | 93.3 | 91.96 | 88 | 95/95 | 162.45 | 70 | 0.0029 |
| 3000000 | 93.5 | 92.46 | 84 | 95/95 | 161.75 | 70 | 0.0029 |
