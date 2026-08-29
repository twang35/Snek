# b21d-beta05seed4

![b21d-beta05seed4 progress](b21d-beta05seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 94.7, perfect games 80%.

Training was resumed at step 165000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | b21d-beta05seed4 |
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
| importance_sampling_beta | 0.4 -> 0.5 over 300000 steps |
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

3001 evals so far. Full series in [`b21d-beta05seed4_evals.json`](b21d-beta05seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 0 | 1/95 | -4.5 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | 0.2 | 0 | 0.4 |
| 2000 | 0.8 | 0.75 | 0 | 5/95 | 0.3 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 93.4 | 93.94 | 84 | 95/95 | 151.25 | 60 | 0.0025 |
| 2990000 | 93.7 | 93.94 | 91 | 95/95 | 141.15 | 50 | 0.0026 |
| 2991000 | 93.9 | 93.84 | 91 | 95/95 | 161.7 | 70 | 0.0026 |
| 2992000 | 94.0 | 93.8 | 91 | 95/95 | 151.4 | 60 | 0.0026 |
| 2993000 | 93.7 | 93.74 | 90 | 95/95 | 141.15 | 50 | 0.0026 |
| 2994000 | 94.8 | 94.02 | 93 | 95/95 | 183.4 | 90 | 0.0026 |
| 2995000 | 93.2 | 93.92 | 89 | 95/95 | 140.2 | 50 | 0.0027 |
| 2996000 | 93.8 | 93.9 | 90 | 95/95 | 141.25 | 50 | 0.0027 |
| 2997000 | 94.4 | 93.98 | 93 | 95/95 | 162.2 | 70 | 0.0027 |
| 2998000 | 93.0 | 93.84 | 91 | 95/95 | 109.25 | 20 | 0.0029 |
| 2999000 | 94.4 | 93.76 | 93 | 95/95 | 162.2 | 70 | 0.0028 |
| 3000000 | 94.7 | 94.06 | 93 | 95/95 | 172.9 | 80 | 0.0028 |
