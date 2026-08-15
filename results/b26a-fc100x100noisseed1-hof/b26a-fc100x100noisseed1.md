# b26a-fc100x100noisseed1

![b26a-fc100x100noisseed1 progress](b26a-fc100x100noisseed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.7, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | b26a-fc100x100noisseed1 |
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
| fc_layer_params | (100, 100) |
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

3001 evals so far. Full series in [`b26a-fc100x100noisseed1_evals.json`](b26a-fc100x100noisseed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.9 | 0.9 | 0 | 3/95 | -4.1 | 0 | 0.4 |
| 1000 | 8.4 | 8.4 | 0 | 22/95 | 6.1 | 0 | 0.4 |
| 2000 | 7.2 | 7.8 | 2 | 14/95 | 6.25 | 0 | 0.2 |
| ... | | | | | | | |
| 2989000 | 95.0 | 94.28 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 2990000 | 94.5 | 94.18 | 90 | 95/95 | 183.55 | 90 | 0.002 |
| 2991000 | 94.5 | 94.08 | 92 | 95/95 | 173.15 | 80 | 0.002 |
| 2992000 | 94.3 | 94.36 | 88 | 95/95 | 183.35 | 90 | 0.002 |
| 2993000 | 93.1 | 94.28 | 82 | 95/95 | 161.8 | 70 | 0.002 |
| 2994000 | 93.2 | 93.92 | 80 | 95/95 | 171.85 | 80 | 0.002 |
| 2995000 | 94.6 | 93.94 | 93 | 95/95 | 172.8 | 80 | 0.002 |
| 2996000 | 93.6 | 93.76 | 91 | 95/95 | 142.4 | 50 | 0.002 |
| 2997000 | 93.4 | 93.58 | 88 | 95/95 | 152.6 | 60 | 0.002 |
| 2998000 | 94.7 | 93.9 | 92 | 95/95 | 183.3 | 90 | 0.002 |
| 2999000 | 93.5 | 93.96 | 88 | 95/95 | 151.8 | 60 | 0.002 |
| 3000000 | 93.7 | 93.78 | 91 | 95/95 | 152.9 | 60 | 0.002 |
