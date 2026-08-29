# b26b-fc100x100noisseed2

![b26b-fc100x100noisseed2 progress](b26b-fc100x100noisseed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.1, perfect games 70%.

## Config

| setting | value |
|---|---|
| policy_name | b26b-fc100x100noisseed2 |
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

3001 evals so far. Full series in [`b26b-fc100x100noisseed2_evals.json`](b26b-fc100x100noisseed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.5 | 0.5 | 0 | 2/95 | 0.0 | 0 | 0.4 |
| 2000 | 0.8 | 0.65 | 0 | 2/95 | 0.3 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.7 | 92.26 | 92 | 95/95 | 183.75 | 90 | 0.0023 |
| 2990000 | 93.1 | 92.02 | 78 | 95/95 | 171.75 | 80 | 0.0023 |
| 2991000 | 94.4 | 93.94 | 91 | 95/95 | 173.05 | 80 | 0.0022 |
| 2992000 | 90.7 | 93.18 | 74 | 95/95 | 130.0 | 40 | 0.0022 |
| 2993000 | 94.0 | 93.38 | 90 | 95/95 | 163.15 | 70 | 0.0023 |
| 2994000 | 93.5 | 93.14 | 80 | 95/95 | 182.55 | 90 | 0.0023 |
| 2995000 | 94.8 | 93.48 | 93 | 95/95 | 183.85 | 90 | 0.0022 |
| 2996000 | 94.8 | 93.56 | 93 | 95/95 | 183.4 | 90 | 0.0022 |
| 2997000 | 93.5 | 94.12 | 82 | 95/95 | 172.6 | 80 | 0.0022 |
| 2998000 | 92.8 | 93.88 | 84 | 95/95 | 161.5 | 70 | 0.0022 |
| 2999000 | 94.1 | 94.0 | 90 | 95/95 | 163.25 | 70 | 0.0022 |
| 3000000 | 93.1 | 93.66 | 88 | 95/95 | 162.25 | 70 | 0.0022 |
