# b23b-beta01seed2

![b23b-beta01seed2 progress](b23b-beta01seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 76.5, perfect games 70%.

## Config

| setting | value |
|---|---|
| policy_name | b23b-beta01seed2 |
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

3001 evals so far. Full series in [`b23b-beta01seed2_evals.json`](b23b-beta01seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.9 | 0.9 | 0 | 3/95 | 0.4 | 0 | 0.4 |
| 2000 | 0.8 | 0.85 | 0 | 2/95 | 0.3 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 89.6 | 92.7 | 56 | 95/95 | 137.95 | 50 | 0.0023 |
| 2990000 | 85.2 | 91.16 | 5 | 95/95 | 164.3 | 80 | 0.0023 |
| 2991000 | 93.9 | 91.46 | 87 | 95/95 | 172.55 | 80 | 0.0023 |
| 2992000 | 91.6 | 90.88 | 64 | 95/95 | 170.7 | 80 | 0.0022 |
| 2993000 | 95.0 | 91.06 | 95 | 95/95 | 194.0 | 100 | 0.0022 |
| 2994000 | 66.2 | 86.38 | 3 | 95/95 | 105.05 | 40 | 0.0022 |
| 2995000 | 92.4 | 87.82 | 69 | 95/95 | 181.45 | 90 | 0.0022 |
| 2996000 | 92.1 | 87.46 | 80 | 95/95 | 171.2 | 80 | 0.0022 |
| 2997000 | 82.3 | 85.6 | 4 | 95/95 | 151.45 | 70 | 0.0022 |
| 2998000 | 95.0 | 85.6 | 95 | 95/95 | 194.0 | 100 | 0.0021 |
| 2999000 | 94.7 | 91.3 | 92 | 95/95 | 183.75 | 90 | 0.0021 |
| 3000000 | 76.5 | 88.12 | 2 | 95/95 | 145.65 | 70 | 0.0021 |
