# b20o-fc320seed3

![b20o-fc320seed3 progress](b20o-fc320seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 243000, avg score 93.1, perfect games 30%.

## Config

| setting | value |
|---|---|
| policy_name | b20o-fc320seed3 |
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
| fc_layer_params | (320,) |
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

244 evals so far. Full series in [`b20o-fc320seed3_evals.json`](b20o-fc320seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 1.4 | 1.4 | 0 | 3/95 | 0.9 | 0 | 0.4 |
| 2000 | 1.4 | 1.4 | 0 | 7/95 | 0.9 | 0 | 0.4 |
| ... | | | | | | | |
| 232000 | 93.0 | 92.34 | 90 | 95/95 | 142.25 | 50 | 0.0053 |
| 233000 | 93.0 | 92.2 | 91 | 95/95 | 122.35 | 30 | 0.0054 |
| 234000 | 94.0 | 92.7 | 90 | 95/95 | 163.15 | 70 | 0.0053 |
| 235000 | 90.7 | 92.4 | 74 | 95/95 | 139.5 | 50 | 0.0052 |
| 236000 | 89.9 | 92.12 | 84 | 95/95 | 109.3 | 20 | 0.0052 |
| 237000 | 92.6 | 92.04 | 88 | 95/95 | 131.9 | 40 | 0.0051 |
| 238000 | 91.1 | 91.66 | 76 | 95/95 | 120.45 | 30 | 0.005 |
| 239000 | 92.6 | 91.38 | 90 | 95/95 | 112.0 | 20 | 0.005 |
| 240000 | 93.5 | 91.94 | 88 | 95/95 | 152.7 | 60 | 0.005 |
| 241000 | 91.0 | 92.16 | 84 | 95/95 | 110.4 | 20 | 0.0051 |
| 242000 | 92.3 | 92.1 | 80 | 95/95 | 151.5 | 60 | 0.0049 |
| 243000 | 93.1 | 92.5 | 92 | 95/95 | 122.0 | 30 | 0.005 |
