# b15c-nstep3seed3

![b15c-nstep3seed3 progress](b15c-nstep3seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 294000, avg score 86.9, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b15c-nstep3seed3 |
| seed | 3 |
| zeroed_observations | none |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.995 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 3 |
| initial_epsilon | 0.4 |
| min_epsilon | 0.002 |
| epsilon_schedule | bootstrap on avg_reward [2, 5, 10, 15, 20] then geometric to floor by 80% trailing-30 perfect |
| guided_fraction | 0.8 |
| exploration_shield | 80% of refinement-phase episodes draw the epsilon move from non-fatal actions; greedy moves never shielded |
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | td_loss |
| importance_sampling_beta | disabled |
| max_steps | 10000000 |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

295 evals so far. Full series in [`b15c-nstep3seed3_evals.json`](b15c-nstep3seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.902 | 0 | 0.4 |
| 1000 | 0.1 | 0.1 | 0 | 1/95 | -0.9 | 0 | 0.4 |
| 2000 | 0.0 | 0.05 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| ... | | | | | | | |
| 283000 | 85.5 | 88.24 | 79 | 95/95 | 104.08 | 20 | 0.0098 |
| 284000 | 83.0 | 87.42 | 60 | 95/95 | 90.805 | 10 | 0.0099 |
| 285000 | 85.3 | 86.92 | 78 | 95/95 | 93.993 | 10 | 0.0099 |
| 286000 | 89.6 | 86.66 | 78 | 95/95 | 116.915 | 30 | 0.0099 |
| 287000 | 86.8 | 86.04 | 81 | 92/95 | 84.967 | 0 | 0.0099 |
| 288000 | 82.8 | 85.5 | 40 | 95/95 | 89.755 | 10 | 0.0099 |
| 289000 | 85.5 | 86.0 | 78 | 92/95 | 83.273 | 0 | 0.0099 |
| 290000 | 86.7 | 86.28 | 81 | 92/95 | 84.893 | 0 | 0.0099 |
| 291000 | 84.2 | 85.2 | 75 | 90/95 | 81.551 | 0 | 0.0099 |
| 292000 | 85.6 | 84.96 | 56 | 94/95 | 82.98 | 0 | 0.01 |
| 293000 | 85.7 | 85.54 | 78 | 94/95 | 83.475 | 0 | 0.01 |
| 294000 | 86.9 | 85.82 | 80 | 93/95 | 85.554 | 0 | 0.0101 |
