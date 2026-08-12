# b22c-noisseed3

![b22c-noisseed3 progress](b22c-noisseed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.7, perfect games 50%.

## Config

| setting | value |
|---|---|
| policy_name | b22c-noisseed3 |
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

3001 evals so far. Full series in [`b22c-noisseed3_evals.json`](b22c-noisseed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 1.1 | 1.1 | 0 | 4/95 | 0.6 | 0 | 0.4 |
| 2000 | 0.8 | 0.95 | 0 | 3/95 | 0.3 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.6 | 90.88 | 93 | 95/95 | 173.7 | 80 | 0.0039 |
| 2990000 | 90.4 | 90.9 | 65 | 95/95 | 127.45 | 40 | 0.0039 |
| 2991000 | 94.4 | 91.86 | 92 | 95/95 | 162.2 | 70 | 0.0039 |
| 2992000 | 82.4 | 90.44 | 2 | 95/95 | 120.8 | 40 | 0.0039 |
| 2993000 | 94.4 | 91.24 | 93 | 95/95 | 162.2 | 70 | 0.0038 |
| 2994000 | 94.4 | 91.2 | 93 | 95/95 | 151.8 | 60 | 0.0038 |
| 2995000 | 93.8 | 91.88 | 91 | 95/95 | 141.7 | 50 | 0.0037 |
| 2996000 | 87.2 | 90.44 | 26 | 95/95 | 134.65 | 50 | 0.0037 |
| 2997000 | 92.5 | 92.46 | 82 | 95/95 | 130.0 | 40 | 0.0037 |
| 2998000 | 94.5 | 92.48 | 93 | 95/95 | 162.3 | 70 | 0.0037 |
| 2999000 | 94.6 | 92.52 | 91 | 95/95 | 183.2 | 90 | 0.0035 |
| 3000000 | 93.7 | 92.5 | 91 | 95/95 | 141.15 | 50 | 0.0035 |
