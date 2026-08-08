# b17b-forkseed2

![b17b-forkseed2 progress](b17b-forkseed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 133000, avg score 87.1, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b17b-forkseed2 |
| seed | 2 |
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

134 evals so far. Full series in [`b17b-forkseed2_evals.json`](b17b-forkseed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.8 | 0.8 | 0 | 3/95 | -0.15 | 0 | 0.4 |
| 2000 | 0.7 | 0.75 | 0 | 2/95 | 0.2 | 0 | 0.4 |
| ... | | | | | | | |
| 122000 | 74.5 | 78.12 | 21 | 92/95 | 74.0 | 0 | 0.0124 |
| 123000 | 88.3 | 83.3 | 82 | 95/95 | 97.75 | 10 | 0.0123 |
| 124000 | 87.6 | 83.68 | 78 | 93/95 | 87.1 | 0 | 0.0123 |
| 125000 | 82.4 | 83.12 | 50 | 95/95 | 91.85 | 10 | 0.0122 |
| 126000 | 83.6 | 83.28 | 66 | 93/95 | 83.1 | 0 | 0.0122 |
| 127000 | 83.7 | 85.12 | 58 | 92/95 | 83.2 | 0 | 0.0122 |
| 128000 | 84.4 | 84.34 | 61 | 91/95 | 83.9 | 0 | 0.0122 |
| 129000 | 86.5 | 84.12 | 78 | 95/95 | 95.95 | 10 | 0.0121 |
| 130000 | 80.7 | 83.78 | 52 | 92/95 | 80.2 | 0 | 0.0121 |
| 131000 | 86.6 | 84.38 | 82 | 90/95 | 86.1 | 0 | 0.0121 |
| 132000 | 89.3 | 85.5 | 84 | 95/95 | 98.75 | 10 | 0.012 |
| 133000 | 87.1 | 86.04 | 84 | 90/95 | 86.6 | 0 | 0.012 |
