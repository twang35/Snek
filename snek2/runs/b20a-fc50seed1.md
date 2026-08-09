# b20a-fc50seed1

![b20a-fc50seed1 progress](b20a-fc50seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2498000, avg score 92.9, perfect games 30%.

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

2499 evals so far. Full series in [`b20a-fc50seed1_evals.json`](b20a-fc50seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -0.5 | 0 | 0.4 |
| 2000 | 0.1 | 0.05 | 0 | 1/95 | -0.4 | 0 | 0.4 |
| ... | | | | | | | |
| 2487000 | 92.1 | 91.2 | 89 | 95/95 | 109.25 | 20 | 0.0072 |
| 2488000 | 92.2 | 91.6 | 90 | 95/95 | 98.95 | 10 | 0.0072 |
| 2489000 | 91.8 | 91.88 | 88 | 95/95 | 99.0 | 10 | 0.0073 |
| 2490000 | 90.2 | 91.46 | 66 | 95/95 | 117.3 | 30 | 0.0073 |
| 2491000 | 92.9 | 91.84 | 91 | 95/95 | 119.55 | 30 | 0.0072 |
| 2492000 | 89.3 | 91.28 | 62 | 95/95 | 96.5 | 10 | 0.0072 |
| 2493000 | 91.4 | 91.12 | 87 | 95/95 | 97.7 | 10 | 0.0074 |
| 2494000 | 92.1 | 91.18 | 86 | 95/95 | 119.65 | 30 | 0.0075 |
| 2495000 | 91.8 | 91.5 | 84 | 95/95 | 97.65 | 10 | 0.0076 |
| 2496000 | 94.1 | 91.74 | 93 | 95/95 | 131.15 | 40 | 0.0076 |
| 2497000 | 93.7 | 92.62 | 91 | 95/95 | 120.35 | 30 | 0.0076 |
| 2498000 | 92.9 | 92.92 | 88 | 95/95 | 120.45 | 30 | 0.0076 |
