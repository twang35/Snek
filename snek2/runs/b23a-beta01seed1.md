# b23a-beta01seed1

![b23a-beta01seed1 progress](b23a-beta01seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 95.0, perfect games 100%.

## Config

| setting | value |
|---|---|
| policy_name | b23a-beta01seed1 |
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

3001 evals so far. Full series in [`b23a-beta01seed1_evals.json`](b23a-beta01seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -0.5 | 0 | 0.4 |
| 2000 | 0.1 | 0.05 | 0 | 1/95 | -0.4 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.6 | 94.08 | 93 | 95/95 | 172.8 | 80 | 0.0024 |
| 2990000 | 94.8 | 94.66 | 93 | 95/95 | 183.4 | 90 | 0.0024 |
| 2991000 | 94.2 | 94.68 | 89 | 95/95 | 172.85 | 80 | 0.0024 |
| 2992000 | 94.2 | 94.56 | 91 | 95/95 | 162.0 | 70 | 0.0024 |
| 2993000 | 94.6 | 94.48 | 93 | 95/95 | 172.8 | 80 | 0.0023 |
| 2994000 | 93.7 | 94.3 | 91 | 95/95 | 141.15 | 50 | 0.0024 |
| 2995000 | 91.3 | 93.6 | 58 | 95/95 | 180.35 | 90 | 0.0023 |
| 2996000 | 94.6 | 93.68 | 93 | 95/95 | 172.8 | 80 | 0.0023 |
| 2997000 | 94.8 | 93.8 | 93 | 95/95 | 183.4 | 90 | 0.0023 |
| 2998000 | 95.0 | 93.88 | 95 | 95/95 | 194.0 | 100 | 0.0022 |
| 2999000 | 95.0 | 94.14 | 95 | 95/95 | 194.0 | 100 | 0.0022 |
| 3000000 | 95.0 | 94.88 | 95 | 95/95 | 194.0 | 100 | 0.0022 |
