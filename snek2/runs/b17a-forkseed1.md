# b17a-forkseed1

![b17a-forkseed1 progress](b17a-forkseed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 110000, avg score 82.5, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b17a-forkseed1 |
| seed | 1 |
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

111 evals so far. Full series in [`b17a-forkseed1_evals.json`](b17a-forkseed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -0.5 | 0 | 0.4 |
| 2000 | 0.0 | 0.0 | 0 | 0/95 | -0.5 | 0 | 0.4 |
| ... | | | | | | | |
| 99000 | 60.1 | 62.38 | 10 | 86/95 | 59.6 | 0 | 0.0125 |
| 100000 | 51.3 | 60.96 | 16 | 88/95 | 50.8 | 0 | 0.0125 |
| 101000 | 52.7 | 56.64 | 11 | 89/95 | 52.2 | 0 | 0.0125 |
| 102000 | 84.9 | 66.6 | 72 | 91/95 | 84.4 | 0 | 0.0125 |
| 103000 | 83.8 | 66.56 | 80 | 88/95 | 83.3 | 0 | 0.0125 |
| 104000 | 82.5 | 71.04 | 68 | 92/95 | 81.55 | 0 | 0.0125 |
| 105000 | 46.0 | 69.98 | 2 | 83/95 | 45.5 | 0 | 0.0125 |
| 106000 | 81.8 | 75.8 | 72 | 91/95 | 80.85 | 0 | 0.0125 |
| 107000 | 84.6 | 75.74 | 79 | 91/95 | 84.1 | 0 | 0.0125 |
| 108000 | 78.3 | 74.64 | 54 | 89/95 | 77.35 | 0 | 0.0125 |
| 109000 | 82.2 | 74.58 | 76 | 88/95 | 81.7 | 0 | 0.0125 |
| 110000 | 82.5 | 81.88 | 70 | 91/95 | 82.0 | 0 | 0.0125 |
