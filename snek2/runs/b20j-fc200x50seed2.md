# b20j-fc200x50seed2

![b20j-fc200x50seed2 progress](b20j-fc200x50seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 710000, avg score 92.1, perfect games 30%.

## Config

| setting | value |
|---|---|
| policy_name | b20j-fc200x50seed2 |
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
| fc_layer_params | (200, 50) |
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

711 evals so far. Full series in [`b20j-fc200x50seed2_evals.json`](b20j-fc200x50seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.9 | 0.9 | 0 | 2/95 | 0.4 | 0 | 0.4 |
| 2000 | 0.7 | 0.8 | 0 | 3/95 | 0.2 | 0 | 0.4 |
| ... | | | | | | | |
| 699000 | 90.8 | 91.34 | 70 | 95/95 | 119.7 | 30 | 0.0067 |
| 700000 | 90.6 | 91.98 | 79 | 95/95 | 99.15 | 10 | 0.0069 |
| 701000 | 90.5 | 92.08 | 76 | 95/95 | 139.75 | 50 | 0.0067 |
| 702000 | 92.2 | 91.7 | 88 | 95/95 | 121.55 | 30 | 0.0068 |
| 703000 | 93.3 | 91.48 | 90 | 95/95 | 122.65 | 30 | 0.0069 |
| 704000 | 91.6 | 91.64 | 86 | 95/95 | 100.15 | 10 | 0.0069 |
| 705000 | 93.4 | 92.2 | 91 | 95/95 | 132.7 | 40 | 0.0068 |
| 706000 | 91.8 | 92.46 | 88 | 95/95 | 110.75 | 20 | 0.0069 |
| 707000 | 92.0 | 92.42 | 84 | 95/95 | 130.85 | 40 | 0.0068 |
| 708000 | 91.0 | 91.96 | 80 | 95/95 | 129.85 | 40 | 0.0067 |
| 709000 | 86.7 | 90.98 | 38 | 95/95 | 95.7 | 10 | 0.0067 |
| 710000 | 92.1 | 90.72 | 84 | 95/95 | 121.0 | 30 | 0.0067 |
