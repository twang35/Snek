# b20ab-fc93x93seed2

![b20ab-fc93x93seed2 progress](b20ab-fc93x93seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 92.9, perfect games 30%.

## Config

| setting | value |
|---|---|
| policy_name | b20ab-fc93x93seed2 |
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
| fc_layer_params | (93, 93) |
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

3001 evals so far. Full series in [`b20ab-fc93x93seed2_evals.json`](b20ab-fc93x93seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 5.3 | 5.3 | 0 | 46/95 | 4.35 | 0 | 0.4 |
| 2000 | 2.2 | 3.75 | 1 | 4/95 | 1.7 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 93.0 | 93.8 | 86 | 95/95 | 129.6 | 40 | 0.0038 |
| 2990000 | 94.1 | 93.9 | 93 | 95/95 | 141.1 | 50 | 0.0038 |
| 2991000 | 93.8 | 93.72 | 91 | 95/95 | 141.25 | 50 | 0.0038 |
| 2992000 | 93.9 | 93.76 | 91 | 95/95 | 130.5 | 40 | 0.0038 |
| 2993000 | 93.5 | 93.66 | 92 | 95/95 | 109.3 | 20 | 0.0039 |
| 2994000 | 93.8 | 93.82 | 91 | 95/95 | 141.25 | 50 | 0.0039 |
| 2995000 | 93.2 | 93.64 | 86 | 95/95 | 140.2 | 50 | 0.004 |
| 2996000 | 92.6 | 93.4 | 87 | 95/95 | 98.0 | 10 | 0.0042 |
| 2997000 | 93.8 | 93.38 | 89 | 95/95 | 140.8 | 50 | 0.0043 |
| 2998000 | 93.2 | 93.32 | 91 | 95/95 | 119.85 | 30 | 0.0044 |
| 2999000 | 94.7 | 93.5 | 93 | 95/95 | 172.9 | 80 | 0.0043 |
| 3000000 | 92.9 | 93.44 | 89 | 95/95 | 119.1 | 30 | 0.0044 |
