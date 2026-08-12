# b23a-beta01seed1

![b23a-beta01seed1 progress](b23a-beta01seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 951000, avg score 94.6, perfect games 80%.

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

952 evals so far. Full series in [`b23a-beta01seed1_evals.json`](b23a-beta01seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -0.5 | 0 | 0.4 |
| 2000 | 0.1 | 0.05 | 0 | 1/95 | -0.4 | 0 | 0.4 |
| ... | | | | | | | |
| 940000 | 94.6 | 94.1 | 93 | 95/95 | 173.25 | 80 | 0.0031 |
| 941000 | 93.8 | 94.02 | 91 | 95/95 | 152.55 | 60 | 0.003 |
| 942000 | 93.8 | 94.08 | 92 | 95/95 | 142.6 | 50 | 0.0031 |
| 943000 | 94.6 | 94.12 | 93 | 95/95 | 173.7 | 80 | 0.003 |
| 944000 | 94.2 | 94.2 | 93 | 95/95 | 153.4 | 60 | 0.0031 |
| 945000 | 93.9 | 94.06 | 88 | 95/95 | 162.6 | 70 | 0.0031 |
| 946000 | 94.3 | 94.16 | 92 | 95/95 | 163.45 | 70 | 0.0031 |
| 947000 | 94.1 | 94.22 | 88 | 95/95 | 173.2 | 80 | 0.003 |
| 948000 | 92.1 | 93.72 | 81 | 95/95 | 131.4 | 40 | 0.0031 |
| 949000 | 94.2 | 93.72 | 93 | 95/95 | 153.4 | 60 | 0.0031 |
| 950000 | 94.8 | 93.9 | 93 | 95/95 | 183.85 | 90 | 0.003 |
| 951000 | 94.6 | 93.96 | 93 | 95/95 | 173.7 | 80 | 0.003 |
