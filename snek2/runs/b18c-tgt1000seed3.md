# b18c-tgt1000seed3

![b18c-tgt1000seed3 progress](b18c-tgt1000seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2510000, avg score 94.8, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b18c-tgt1000seed3 |
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

2511 evals so far. Full series in [`b18c-tgt1000seed3_evals.json`](b18c-tgt1000seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 1.0 | 1.0 | 0 | 4/95 | 0.5 | 0 | 0.4 |
| 2000 | 1.1 | 1.05 | 0 | 5/95 | 0.6 | 0 | 0.4 |
| ... | | | | | | | |
| 2499000 | 94.8 | 94.64 | 93 | 95/95 | 183.4 | 90 | 0.0022 |
| 2500000 | 94.6 | 94.6 | 93 | 95/95 | 172.8 | 80 | 0.0021 |
| 2501000 | 94.8 | 94.64 | 93 | 95/95 | 183.4 | 90 | 0.0021 |
| 2502000 | 95.0 | 94.76 | 95 | 95/95 | 194.0 | 100 | 0.002 |
| 2503000 | 93.9 | 94.62 | 90 | 95/95 | 151.75 | 60 | 0.0021 |
| 2504000 | 94.4 | 94.54 | 93 | 95/95 | 162.2 | 70 | 0.0021 |
| 2505000 | 94.8 | 94.58 | 93 | 95/95 | 183.4 | 90 | 0.0021 |
| 2506000 | 94.6 | 94.54 | 93 | 95/95 | 172.8 | 80 | 0.0021 |
| 2507000 | 94.6 | 94.46 | 93 | 95/95 | 172.8 | 80 | 0.0021 |
| 2508000 | 94.2 | 94.52 | 91 | 95/95 | 162.0 | 70 | 0.0021 |
| 2509000 | 94.6 | 94.56 | 93 | 95/95 | 172.8 | 80 | 0.0021 |
| 2510000 | 94.8 | 94.56 | 93 | 95/95 | 183.4 | 90 | 0.002 |
