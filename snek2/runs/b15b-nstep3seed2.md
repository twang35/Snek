# b15b-nstep3seed2

![b15b-nstep3seed2 progress](b15b-nstep3seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 287000, avg score 89.0, perfect games 20%.

## Config

| setting | value |
|---|---|
| policy_name | b15b-nstep3seed2 |
| seed | 2 |
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

288 evals so far. Full series in [`b15b-nstep3seed2_evals.json`](b15b-nstep3seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.003 | 0 | 0.4 |
| 1000 | 0.1 | 0.1 | 0 | 1/95 | -4.903 | 0 | 0.4 |
| 2000 | 0.2 | 0.15 | 0 | 1/95 | -2.14 | 0 | 0.4 |
| ... | | | | | | | |
| 276000 | 86.6 | 88.4 | 80 | 94/95 | 84.64 | 0 | 0.0111 |
| 277000 | 79.5 | 86.28 | 14 | 90/95 | 78.146 | 0 | 0.0111 |
| 278000 | 88.0 | 86.36 | 66 | 95/95 | 115.927 | 30 | 0.0108 |
| 279000 | 83.6 | 85.26 | 14 | 95/95 | 111.656 | 30 | 0.0106 |
| 280000 | 80.6 | 83.66 | 51 | 95/95 | 89.182 | 10 | 0.0105 |
| 281000 | 87.9 | 83.92 | 75 | 93/95 | 85.93 | 0 | 0.0106 |
| 282000 | 88.1 | 85.64 | 82 | 95/95 | 95.166 | 10 | 0.0105 |
| 283000 | 85.8 | 85.2 | 58 | 95/95 | 93.354 | 10 | 0.0105 |
| 284000 | 81.0 | 84.68 | 14 | 95/95 | 89.218 | 10 | 0.0106 |
| 285000 | 88.4 | 86.24 | 84 | 95/95 | 96.807 | 10 | 0.0105 |
| 286000 | 86.5 | 85.96 | 78 | 95/95 | 94.179 | 10 | 0.0106 |
| 287000 | 89.0 | 86.14 | 72 | 95/95 | 106.14 | 20 | 0.0104 |
