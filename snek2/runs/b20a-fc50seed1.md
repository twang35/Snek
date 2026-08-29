# b20a-fc50seed1

![b20a-fc50seed1 progress](b20a-fc50seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.2, perfect games 20%.

Training was resumed at step 2498000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | b20a-fc50seed1 |
| seed | 1 |
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

3001 evals so far. Full series in [`b20a-fc50seed1_evals.json`](b20a-fc50seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -0.5 | 0 | 0.4 |
| 2000 | 0.1 | 0.05 | 0 | 1/95 | -0.4 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 92.9 | 90.48 | 91 | 95/95 | 108.7 | 20 | 0.0069 |
| 2990000 | 91.8 | 90.3 | 87 | 95/95 | 97.2 | 10 | 0.0072 |
| 2991000 | 93.3 | 91.28 | 91 | 95/95 | 109.1 | 20 | 0.0072 |
| 2992000 | 93.1 | 92.88 | 89 | 95/95 | 129.7 | 40 | 0.0071 |
| 2993000 | 93.6 | 92.94 | 91 | 95/95 | 109.4 | 20 | 0.0071 |
| 2994000 | 92.5 | 92.86 | 85 | 95/95 | 119.15 | 30 | 0.0069 |
| 2995000 | 93.2 | 93.14 | 91 | 95/95 | 119.85 | 30 | 0.0069 |
| 2996000 | 88.5 | 92.18 | 50 | 94/95 | 83.5 | 0 | 0.0071 |
| 2997000 | 91.6 | 91.88 | 89 | 95/95 | 107.4 | 20 | 0.0071 |
| 2998000 | 90.3 | 91.22 | 68 | 95/95 | 95.7 | 10 | 0.0072 |
| 2999000 | 89.2 | 90.56 | 70 | 95/95 | 105.45 | 20 | 0.0073 |
| 3000000 | 93.2 | 90.56 | 91 | 95/95 | 109.0 | 20 | 0.0074 |
