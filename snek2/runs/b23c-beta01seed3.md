# b23c-beta01seed3

![b23c-beta01seed3 progress](b23c-beta01seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 94.1, perfect games 50%.

## Config

| setting | value |
|---|---|
| policy_name | b23c-beta01seed3 |
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

3001 evals so far. Full series in [`b23c-beta01seed3_evals.json`](b23c-beta01seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 1.0 | 1.0 | 0 | 4/95 | 0.5 | 0 | 0.4 |
| 2000 | 1.0 | 1.0 | 0 | 3/95 | 0.5 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.0 | 93.8 | 89 | 95/95 | 162.25 | 70 | 0.0024 |
| 2990000 | 94.8 | 94.0 | 93 | 95/95 | 183.4 | 90 | 0.0024 |
| 2991000 | 94.8 | 94.34 | 93 | 95/95 | 183.4 | 90 | 0.0024 |
| 2992000 | 92.7 | 93.94 | 76 | 95/95 | 160.95 | 70 | 0.0024 |
| 2993000 | 94.6 | 94.18 | 93 | 95/95 | 172.8 | 80 | 0.0024 |
| 2994000 | 94.6 | 94.3 | 93 | 95/95 | 172.8 | 80 | 0.0023 |
| 2995000 | 94.8 | 94.3 | 93 | 95/95 | 183.4 | 90 | 0.0023 |
| 2996000 | 93.6 | 94.06 | 91 | 95/95 | 141.5 | 50 | 0.0023 |
| 2997000 | 93.3 | 94.18 | 78 | 95/95 | 182.35 | 90 | 0.0023 |
| 2998000 | 94.0 | 94.06 | 91 | 95/95 | 151.85 | 60 | 0.0024 |
| 2999000 | 94.4 | 94.02 | 92 | 95/95 | 173.5 | 80 | 0.0024 |
| 3000000 | 94.1 | 93.88 | 93 | 95/95 | 141.1 | 50 | 0.0024 |
