# b17d-forkseed4

![b17d-forkseed4 progress](b17d-forkseed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1514000, avg score 93.5, perfect games 70%.

## Config

| setting | value |
|---|---|
| policy_name | b17d-forkseed4 |
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

1515 evals so far. Full series in [`b17d-forkseed4_evals.json`](b17d-forkseed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 0 | 1/95 | -4.5 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | 0.2 | 0 | 0.4 |
| 2000 | 0.7 | 0.7 | 0 | 2/95 | 0.2 | 0 | 0.4 |
| ... | | | | | | | |
| 1503000 | 94.6 | 93.68 | 93 | 95/95 | 172.8 | 80 | 0.0025 |
| 1504000 | 89.7 | 92.7 | 75 | 95/95 | 148.45 | 60 | 0.0025 |
| 1505000 | 73.5 | 88.84 | 4 | 95/95 | 132.25 | 60 | 0.0026 |
| 1506000 | 94.6 | 89.36 | 93 | 95/95 | 172.8 | 80 | 0.0026 |
| 1507000 | 93.5 | 89.18 | 90 | 95/95 | 130.55 | 40 | 0.0026 |
| 1508000 | 92.6 | 88.78 | 77 | 95/95 | 160.85 | 70 | 0.0027 |
| 1509000 | 93.5 | 89.54 | 84 | 95/95 | 161.3 | 70 | 0.0027 |
| 1510000 | 92.5 | 93.34 | 80 | 95/95 | 139.95 | 50 | 0.0027 |
| 1511000 | 94.0 | 93.22 | 91 | 95/95 | 151.4 | 60 | 0.0028 |
| 1512000 | 93.1 | 93.14 | 80 | 95/95 | 161.35 | 70 | 0.0027 |
| 1513000 | 94.8 | 93.58 | 93 | 95/95 | 183.4 | 90 | 0.0026 |
| 1514000 | 93.5 | 93.58 | 87 | 95/95 | 161.75 | 70 | 0.0026 |
