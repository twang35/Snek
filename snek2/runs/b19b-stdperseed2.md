# b19b-stdperseed2

![b19b-stdperseed2 progress](b19b-stdperseed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2116000, avg score 90.3, perfect games 50%.

## Config

| setting | value |
|---|---|
| policy_name | b19b-stdperseed2 |
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
| importance_sampling_beta | 0.4 -> 1.0 over 1000000 steps |
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

2117 evals so far. Full series in [`b19b-stdperseed2_evals.json`](b19b-stdperseed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.9 | 0.9 | 0 | 3/95 | 0.4 | 0 | 0.4 |
| 2000 | 1.0 | 0.95 | 0 | 3/95 | 0.5 | 0 | 0.4 |
| ... | | | | | | | |
| 2105000 | 94.0 | 92.0 | 92 | 95/95 | 152.75 | 60 | 0.0038 |
| 2106000 | 93.3 | 92.04 | 92 | 95/95 | 120.85 | 30 | 0.0039 |
| 2107000 | 94.1 | 91.92 | 92 | 95/95 | 152.85 | 60 | 0.0039 |
| 2108000 | 94.0 | 91.94 | 91 | 95/95 | 162.7 | 70 | 0.0037 |
| 2109000 | 92.8 | 93.64 | 86 | 95/95 | 132.1 | 40 | 0.0038 |
| 2110000 | 94.3 | 93.7 | 92 | 95/95 | 162.55 | 70 | 0.0037 |
| 2111000 | 92.4 | 93.52 | 88 | 95/95 | 119.95 | 30 | 0.0037 |
| 2112000 | 89.9 | 92.68 | 70 | 95/95 | 148.65 | 60 | 0.0037 |
| 2113000 | 93.5 | 92.58 | 91 | 95/95 | 131.9 | 40 | 0.0038 |
| 2114000 | 92.5 | 92.52 | 84 | 95/95 | 151.25 | 60 | 0.0037 |
| 2115000 | 92.8 | 92.22 | 82 | 95/95 | 141.6 | 50 | 0.0038 |
| 2116000 | 90.3 | 91.8 | 76 | 95/95 | 138.65 | 50 | 0.0039 |
