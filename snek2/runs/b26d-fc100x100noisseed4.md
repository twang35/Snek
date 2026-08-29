# b26d-fc100x100noisseed4

![b26d-fc100x100noisseed4 progress](b26d-fc100x100noisseed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 92.6, perfect games 20%.

## Config

| setting | value |
|---|---|
| policy_name | b26d-fc100x100noisseed4 |
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

3001 evals so far. Full series in [`b26d-fc100x100noisseed4_evals.json`](b26d-fc100x100noisseed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.6 | 0.6 | 0 | 3/95 | 0.1 | 0 | 0.4 |
| 2000 | 0.8 | 0.7 | 0 | 3/95 | 0.3 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.2 | 91.68 | 91 | 95/95 | 162.45 | 70 | 0.0038 |
| 2990000 | 89.5 | 90.78 | 62 | 95/95 | 106.2 | 20 | 0.0038 |
| 2991000 | 93.6 | 90.8 | 89 | 95/95 | 151.0 | 60 | 0.0037 |
| 2992000 | 88.9 | 89.98 | 42 | 95/95 | 136.35 | 50 | 0.0037 |
| 2993000 | 93.8 | 92.0 | 91 | 95/95 | 140.8 | 50 | 0.0037 |
| 2994000 | 92.5 | 91.66 | 82 | 95/95 | 129.55 | 40 | 0.0037 |
| 2995000 | 94.4 | 92.64 | 89 | 95/95 | 183.45 | 90 | 0.0037 |
| 2996000 | 91.4 | 92.2 | 77 | 95/95 | 108.1 | 20 | 0.0037 |
| 2997000 | 93.4 | 93.1 | 91 | 95/95 | 130.0 | 40 | 0.0038 |
| 2998000 | 92.5 | 92.84 | 88 | 95/95 | 119.15 | 30 | 0.0039 |
| 2999000 | 90.8 | 92.5 | 80 | 95/95 | 128.3 | 40 | 0.004 |
| 3000000 | 92.6 | 92.14 | 89 | 95/95 | 108.4 | 20 | 0.0041 |
