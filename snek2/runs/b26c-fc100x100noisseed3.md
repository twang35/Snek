# b26c-fc100x100noisseed3

![b26c-fc100x100noisseed3 progress](b26c-fc100x100noisseed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 94.5, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b26c-fc100x100noisseed3 |
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

3001 evals so far. Full series in [`b26c-fc100x100noisseed3_evals.json`](b26c-fc100x100noisseed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 1.0 | 1.0 | 0 | 3/95 | 0.5 | 0 | 0.4 |
| 2000 | 0.4 | 0.7 | 0 | 1/95 | -0.1 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 69.1 | 71.1 | 0 | 95/95 | 138.25 | 70 | 0.0021 |
| 2990000 | 76.1 | 67.32 | 1 | 95/95 | 145.25 | 70 | 0.0022 |
| 2991000 | 48.2 | 65.48 | 0 | 95/95 | 97.45 | 50 | 0.0022 |
| 2992000 | 48.2 | 61.66 | 0 | 95/95 | 87.5 | 40 | 0.0022 |
| 2993000 | 74.6 | 63.24 | 1 | 95/95 | 133.8 | 60 | 0.0023 |
| 2994000 | 94.3 | 68.28 | 92 | 95/95 | 163.45 | 70 | 0.0023 |
| 2995000 | 93.5 | 71.76 | 84 | 95/95 | 162.65 | 70 | 0.0023 |
| 2996000 | 85.7 | 79.26 | 2 | 95/95 | 174.75 | 90 | 0.0023 |
| 2997000 | 94.8 | 88.58 | 93 | 95/95 | 183.85 | 90 | 0.0022 |
| 2998000 | 92.3 | 92.12 | 68 | 95/95 | 181.35 | 90 | 0.0022 |
| 2999000 | 92.8 | 91.82 | 80 | 95/95 | 161.95 | 70 | 0.0022 |
| 3000000 | 94.5 | 92.02 | 90 | 95/95 | 183.55 | 90 | 0.0022 |
