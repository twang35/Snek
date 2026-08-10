# b20s-fc25x50x25seed3

![b20s-fc25x50x25seed3 progress](b20s-fc25x50x25seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.1, perfect games 20%.

## Config

| setting | value |
|---|---|
| policy_name | b20s-fc25x50x25seed3 |
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
| fc_layer_params | (25, 50, 25) |
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

3001 evals so far. Full series in [`b20s-fc25x50x25seed3_evals.json`](b20s-fc25x50x25seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 0.5 | 0.5 | 0 | 2/95 | 0.0 | 0 | 0.4 |
| 2000 | 0.2 | 0.35 | 0 | 1/95 | -0.3 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 92.8 | 91.3 | 90 | 95/95 | 102.25 | 10 | 0.0065 |
| 2990000 | 92.8 | 91.54 | 86 | 95/95 | 121.7 | 30 | 0.0065 |
| 2991000 | 92.0 | 91.94 | 84 | 94/95 | 90.6 | 0 | 0.0066 |
| 2992000 | 92.4 | 92.32 | 90 | 95/95 | 121.3 | 30 | 0.0065 |
| 2993000 | 88.8 | 91.76 | 71 | 94/95 | 87.85 | 0 | 0.0066 |
| 2994000 | 92.0 | 91.6 | 90 | 95/95 | 100.55 | 10 | 0.0067 |
| 2995000 | 92.8 | 91.6 | 91 | 95/95 | 121.25 | 30 | 0.0069 |
| 2996000 | 91.2 | 91.44 | 80 | 95/95 | 99.3 | 10 | 0.0069 |
| 2997000 | 93.6 | 91.68 | 91 | 95/95 | 142.85 | 50 | 0.007 |
| 2998000 | 92.4 | 92.4 | 89 | 95/95 | 110.9 | 20 | 0.0071 |
| 2999000 | 92.8 | 92.56 | 87 | 95/95 | 121.7 | 30 | 0.0072 |
| 3000000 | 93.1 | 92.62 | 90 | 95/95 | 112.5 | 20 | 0.0073 |
