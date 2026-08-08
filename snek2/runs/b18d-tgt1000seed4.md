# b18d-tgt1000seed4

![b18d-tgt1000seed4 progress](b18d-tgt1000seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2597000, avg score 90.6, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b18d-tgt1000seed4 |
| seed | 4 |
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

2598 evals so far. Full series in [`b18d-tgt1000seed4_evals.json`](b18d-tgt1000seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 0 | 1/95 | -4.5 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | 0.2 | 0 | 0.4 |
| 2000 | 0.4 | 0.55 | 0 | 2/95 | -0.1 | 0 | 0.4 |
| ... | | | | | | | |
| 2586000 | 94.8 | 93.04 | 93 | 95/95 | 183.4 | 90 | 0.002 |
| 2587000 | 94.6 | 93.78 | 91 | 95/95 | 183.65 | 90 | 0.002 |
| 2588000 | 94.8 | 94.66 | 93 | 95/95 | 183.4 | 90 | 0.002 |
| 2589000 | 91.0 | 93.96 | 84 | 95/95 | 129.4 | 40 | 0.002 |
| 2590000 | 94.0 | 93.84 | 86 | 95/95 | 172.65 | 80 | 0.002 |
| 2591000 | 92.1 | 93.3 | 75 | 95/95 | 160.35 | 70 | 0.002 |
| 2592000 | 91.8 | 92.74 | 70 | 95/95 | 170.45 | 80 | 0.002 |
| 2593000 | 94.5 | 92.68 | 93 | 95/95 | 162.3 | 70 | 0.002 |
| 2594000 | 94.1 | 93.3 | 90 | 95/95 | 172.75 | 80 | 0.002 |
| 2595000 | 93.0 | 93.1 | 82 | 95/95 | 161.7 | 70 | 0.002 |
| 2596000 | 92.7 | 93.22 | 83 | 95/95 | 151.0 | 60 | 0.0021 |
| 2597000 | 90.6 | 92.98 | 51 | 95/95 | 179.65 | 90 | 0.002 |
