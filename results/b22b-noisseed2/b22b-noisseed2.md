# b22b-noisseed2

![b22b-noisseed2 progress](b22b-noisseed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 90.6, perfect games 50%.

## Config

| setting | value |
|---|---|
| policy_name | b22b-noisseed2 |
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

3001 evals so far. Full series in [`b22b-noisseed2_evals.json`](b22b-noisseed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.9 | 0.9 | 0 | 3/95 | 0.4 | 0 | 0.4 |
| 2000 | 1.3 | 1.1 | 0 | 4/95 | 0.8 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 92.4 | 93.64 | 78 | 95/95 | 161.1 | 70 | 0.003 |
| 2990000 | 90.8 | 92.8 | 57 | 95/95 | 159.05 | 70 | 0.003 |
| 2991000 | 93.6 | 92.52 | 84 | 95/95 | 171.8 | 80 | 0.0029 |
| 2992000 | 93.2 | 92.78 | 84 | 95/95 | 161.9 | 70 | 0.0028 |
| 2993000 | 93.5 | 92.7 | 88 | 95/95 | 152.25 | 60 | 0.0028 |
| 2994000 | 93.8 | 92.98 | 91 | 95/95 | 152.1 | 60 | 0.0028 |
| 2995000 | 87.2 | 92.26 | 34 | 95/95 | 124.7 | 40 | 0.0029 |
| 2996000 | 92.9 | 92.12 | 90 | 95/95 | 120.9 | 30 | 0.003 |
| 2997000 | 93.4 | 92.16 | 91 | 95/95 | 130.45 | 40 | 0.003 |
| 2998000 | 94.8 | 92.42 | 93 | 95/95 | 183.4 | 90 | 0.003 |
| 2999000 | 87.9 | 91.24 | 26 | 95/95 | 166.55 | 80 | 0.0029 |
| 3000000 | 90.6 | 91.92 | 74 | 95/95 | 138.95 | 50 | 0.0029 |
