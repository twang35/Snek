# b23d-beta01seed4

![b23d-beta01seed4 progress](b23d-beta01seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 94.0, perfect games 50%.

## Config

| setting | value |
|---|---|
| policy_name | b23d-beta01seed4 |
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
| importance_sampling_beta | 0.0 -> 0.1 over 300000 steps |
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

3001 evals so far. Full series in [`b23d-beta01seed4_evals.json`](b23d-beta01seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 0 | 1/95 | -4.5 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | 0.2 | 0 | 0.4 |
| 2000 | 0.4 | 0.55 | 0 | 2/95 | -0.1 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.1 | 93.52 | 90 | 95/95 | 162.35 | 70 | 0.0021 |
| 2990000 | 94.8 | 93.56 | 93 | 95/95 | 183.4 | 90 | 0.002 |
| 2991000 | 94.8 | 93.6 | 93 | 95/95 | 183.4 | 90 | 0.002 |
| 2992000 | 92.3 | 93.72 | 70 | 95/95 | 170.95 | 80 | 0.002 |
| 2993000 | 91.6 | 93.52 | 74 | 95/95 | 149.9 | 60 | 0.0021 |
| 2994000 | 90.5 | 92.8 | 66 | 95/95 | 128.45 | 40 | 0.0021 |
| 2995000 | 94.6 | 92.76 | 93 | 95/95 | 172.8 | 80 | 0.0021 |
| 2996000 | 94.4 | 92.68 | 93 | 95/95 | 162.2 | 70 | 0.0022 |
| 2997000 | 94.0 | 93.02 | 91 | 95/95 | 151.4 | 60 | 0.0022 |
| 2998000 | 94.7 | 93.64 | 93 | 95/95 | 172.9 | 80 | 0.0022 |
| 2999000 | 90.5 | 93.64 | 68 | 95/95 | 159.2 | 70 | 0.0022 |
| 3000000 | 94.0 | 93.52 | 93 | 95/95 | 141.0 | 50 | 0.0023 |
