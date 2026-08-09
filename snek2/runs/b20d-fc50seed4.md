# b20d-fc50seed4

![b20d-fc50seed4 progress](b20d-fc50seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2734000, avg score 92.1, perfect games 30%.

## Config

| setting | value |
|---|---|
| policy_name | b20d-fc50seed4 |
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
| importance_sampling_beta | 0.4 -> 1.0 over 300000 steps |
| max_steps | 10000000 |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.0 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2735 evals so far. Full series in [`b20d-fc50seed4_evals.json`](b20d-fc50seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 0 | 1/95 | -4.5 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | 0.2 | 0 | 0.4 |
| 2000 | 1.6 | 1.15 | 0 | 6/95 | 1.1 | 0 | 0.4 |
| ... | | | | | | | |
| 2723000 | 93.6 | 93.58 | 89 | 95/95 | 141.95 | 50 | 0.0045 |
| 2724000 | 93.3 | 93.48 | 86 | 95/95 | 142.55 | 50 | 0.0044 |
| 2725000 | 94.5 | 93.58 | 93 | 95/95 | 163.2 | 70 | 0.0043 |
| 2726000 | 94.0 | 93.72 | 91 | 95/95 | 152.3 | 60 | 0.0042 |
| 2727000 | 92.6 | 93.6 | 87 | 95/95 | 100.25 | 10 | 0.0042 |
| 2728000 | 91.7 | 93.22 | 81 | 95/95 | 130.55 | 40 | 0.0042 |
| 2729000 | 90.1 | 92.58 | 74 | 95/95 | 118.1 | 30 | 0.0043 |
| 2730000 | 93.6 | 92.4 | 91 | 95/95 | 141.5 | 50 | 0.0043 |
| 2731000 | 93.6 | 92.32 | 91 | 95/95 | 131.55 | 40 | 0.0043 |
| 2732000 | 93.4 | 92.48 | 91 | 95/95 | 131.35 | 40 | 0.0043 |
| 2733000 | 83.7 | 90.88 | 48 | 95/95 | 92.25 | 10 | 0.0043 |
| 2734000 | 92.1 | 91.28 | 80 | 95/95 | 119.2 | 30 | 0.0043 |
