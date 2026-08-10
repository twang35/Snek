# b20t-fc25x50x25seed4

![b20t-fc25x50x25seed4 progress](b20t-fc25x50x25seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.7, perfect games 50%.

## Config

| setting | value |
|---|---|
| policy_name | b20t-fc25x50x25seed4 |
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
| fc_layer_params | (25, 50, 25) |
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

3001 evals so far. Full series in [`b20t-fc25x50x25seed4_evals.json`](b20t-fc25x50x25seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.3 | 0.3 | 0 | 1/95 | -4.7 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 1/95 | -5.0 | 0 | 0.4 |
| 2000 | 0.9 | 0.45 | 0 | 3/95 | 0.4 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 93.5 | 93.74 | 88 | 95/95 | 141.85 | 50 | 0.0044 |
| 2990000 | 94.0 | 93.7 | 91 | 95/95 | 152.3 | 60 | 0.0043 |
| 2991000 | 94.2 | 93.82 | 93 | 95/95 | 153.4 | 60 | 0.0043 |
| 2992000 | 93.8 | 93.8 | 91 | 95/95 | 152.55 | 60 | 0.0042 |
| 2993000 | 92.7 | 93.64 | 86 | 95/95 | 122.05 | 30 | 0.0043 |
| 2994000 | 92.1 | 93.36 | 81 | 95/95 | 111.05 | 20 | 0.0044 |
| 2995000 | 93.5 | 93.26 | 89 | 95/95 | 142.75 | 50 | 0.0043 |
| 2996000 | 94.0 | 93.22 | 91 | 95/95 | 162.25 | 70 | 0.0043 |
| 2997000 | 93.9 | 93.24 | 93 | 95/95 | 131.4 | 40 | 0.0043 |
| 2998000 | 92.6 | 93.22 | 88 | 95/95 | 120.6 | 30 | 0.0044 |
| 2999000 | 94.0 | 93.6 | 91 | 95/95 | 162.7 | 70 | 0.0043 |
| 3000000 | 93.7 | 93.64 | 90 | 95/95 | 142.05 | 50 | 0.0043 |
