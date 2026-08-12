# b23c-beta01seed3

![b23c-beta01seed3 progress](b23c-beta01seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 933000, avg score 93.2, perfect games 40%.

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

934 evals so far. Full series in [`b23c-beta01seed3_evals.json`](b23c-beta01seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 1.0 | 1.0 | 0 | 4/95 | 0.5 | 0 | 0.4 |
| 2000 | 1.0 | 1.0 | 0 | 3/95 | 0.5 | 0 | 0.4 |
| ... | | | | | | | |
| 922000 | 81.7 | 91.24 | 28 | 95/95 | 111.05 | 30 | 0.0031 |
| 923000 | 93.1 | 91.02 | 82 | 95/95 | 151.4 | 60 | 0.003 |
| 924000 | 93.5 | 90.84 | 86 | 95/95 | 152.7 | 60 | 0.003 |
| 925000 | 94.6 | 91.4 | 93 | 95/95 | 173.7 | 80 | 0.003 |
| 926000 | 94.0 | 91.38 | 93 | 95/95 | 142.35 | 50 | 0.003 |
| 927000 | 93.8 | 93.8 | 87 | 95/95 | 162.5 | 70 | 0.003 |
| 928000 | 77.9 | 90.76 | 4 | 95/95 | 126.7 | 50 | 0.003 |
| 929000 | 92.2 | 90.5 | 81 | 95/95 | 110.7 | 20 | 0.0031 |
| 930000 | 94.4 | 90.46 | 93 | 95/95 | 163.55 | 70 | 0.003 |
| 931000 | 86.6 | 88.98 | 24 | 95/95 | 134.5 | 50 | 0.0031 |
| 932000 | 93.6 | 88.94 | 91 | 95/95 | 142.4 | 50 | 0.0032 |
| 933000 | 93.2 | 92.0 | 87 | 95/95 | 131.6 | 40 | 0.0033 |
