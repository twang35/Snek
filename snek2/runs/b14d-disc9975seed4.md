# b14d-disc9975seed4

![b14d-disc9975seed4 progress](b14d-disc9975seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1483000, avg score 94.5, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b14d-disc9975seed4 |
| seed | 4 |
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
| exploration_shield | 80% of refinement-phase episodes draw the epsilon move from non-fatal actions; greedy moves never shielded |
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | td_loss |
| importance_sampling_beta | disabled |
| max_steps | 5000000 |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

1484 evals so far. Full series in [`b14d-disc9975seed4_evals.json`](b14d-disc9975seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 0 | 1/95 | -4.503 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | 0.147 | 0 | 0.4 |
| 2000 | 0.7 | 0.7 | 0 | 2/95 | 0.146 | 0 | 0.4 |
| ... | | | | | | | |
| 1472000 | 94.5 | 94.04 | 92 | 95/95 | 172.996 | 80 | 0.0024 |
| 1473000 | 93.9 | 94.04 | 90 | 95/95 | 161.971 | 70 | 0.0024 |
| 1474000 | 94.5 | 94.04 | 90 | 95/95 | 183.031 | 90 | 0.0024 |
| 1475000 | 94.2 | 94.32 | 90 | 95/95 | 172.609 | 80 | 0.0024 |
| 1476000 | 93.4 | 94.1 | 88 | 95/95 | 141.469 | 50 | 0.0024 |
| 1477000 | 94.0 | 94.0 | 92 | 95/95 | 152.101 | 60 | 0.0024 |
| 1478000 | 94.5 | 94.12 | 92 | 95/95 | 172.579 | 80 | 0.0024 |
| 1479000 | 94.1 | 94.04 | 92 | 95/95 | 162.135 | 70 | 0.0024 |
| 1480000 | 94.6 | 94.12 | 93 | 95/95 | 173.044 | 80 | 0.0024 |
| 1481000 | 94.5 | 94.34 | 90 | 95/95 | 183.03 | 90 | 0.0023 |
| 1482000 | 94.4 | 94.42 | 93 | 95/95 | 152.04 | 60 | 0.0023 |
| 1483000 | 94.5 | 94.42 | 90 | 95/95 | 182.911 | 90 | 0.0023 |
