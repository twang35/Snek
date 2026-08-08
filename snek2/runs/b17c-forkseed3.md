# b17c-forkseed3

![b17c-forkseed3 progress](b17c-forkseed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1520000, avg score 95.0, perfect games 100%.

## Config

| setting | value |
|---|---|
| policy_name | b17c-forkseed3 |
| seed | 3 |
| zeroed_observations | none |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.9975 |
| target_update_period | 8 |
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
| priority_signal | td_loss |
| importance_sampling_beta | disabled |
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

1521 evals so far. Full series in [`b17c-forkseed3_evals.json`](b17c-forkseed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 2.1 | 2.1 | 0 | 4/95 | 1.6 | 0 | 0.4 |
| 2000 | 0.8 | 1.45 | 0 | 5/95 | 0.3 | 0 | 0.4 |
| ... | | | | | | | |
| 1509000 | 94.8 | 81.96 | 93 | 95/95 | 183.4 | 90 | 0.0026 |
| 1510000 | 87.8 | 82.74 | 27 | 95/95 | 156.05 | 70 | 0.0026 |
| 1511000 | 88.4 | 81.52 | 29 | 95/95 | 177.45 | 90 | 0.0026 |
| 1512000 | 94.8 | 91.98 | 93 | 95/95 | 183.4 | 90 | 0.0025 |
| 1513000 | 92.9 | 91.74 | 80 | 95/95 | 150.75 | 60 | 0.0024 |
| 1514000 | 94.8 | 91.74 | 93 | 95/95 | 183.4 | 90 | 0.0024 |
| 1515000 | 93.7 | 92.92 | 82 | 95/95 | 182.75 | 90 | 0.0024 |
| 1516000 | 90.4 | 93.32 | 51 | 95/95 | 169.05 | 80 | 0.0023 |
| 1517000 | 93.9 | 93.14 | 90 | 95/95 | 151.75 | 60 | 0.0023 |
| 1518000 | 88.2 | 92.2 | 29 | 95/95 | 166.85 | 80 | 0.0023 |
| 1519000 | 93.0 | 91.84 | 79 | 95/95 | 161.25 | 70 | 0.0023 |
| 1520000 | 95.0 | 92.1 | 95 | 95/95 | 194.0 | 100 | 0.0023 |
