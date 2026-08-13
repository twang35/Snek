# b24d-fc320noisseed4

![b24d-fc320noisseed4 progress](b24d-fc320noisseed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 89.7, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | b24d-fc320noisseed4 |
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
| fc_layer_params | (320,) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | td_error |
| importance_sampling_beta | disabled |
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

3001 evals so far. Full series in [`b24d-fc320noisseed4_evals.json`](b24d-fc320noisseed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 2/95 | -0.1 | 0 | 0.4 |
| 2000 | 2.7 | 1.55 | 1 | 9/95 | 2.2 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 89.9 | 91.46 | 76 | 95/95 | 159.05 | 70 | 0.003 |
| 2990000 | 88.8 | 91.08 | 75 | 95/95 | 127.65 | 40 | 0.003 |
| 2991000 | 93.9 | 91.18 | 90 | 95/95 | 162.6 | 70 | 0.003 |
| 2992000 | 91.1 | 90.94 | 79 | 95/95 | 140.35 | 50 | 0.003 |
| 2993000 | 93.2 | 91.38 | 79 | 95/95 | 171.85 | 80 | 0.003 |
| 2994000 | 93.0 | 92.0 | 84 | 95/95 | 162.15 | 70 | 0.003 |
| 2995000 | 89.0 | 92.04 | 72 | 95/95 | 138.25 | 50 | 0.003 |
| 2996000 | 92.5 | 91.76 | 82 | 95/95 | 151.7 | 60 | 0.0031 |
| 2997000 | 88.6 | 91.26 | 75 | 95/95 | 117.95 | 30 | 0.0031 |
| 2998000 | 94.1 | 91.44 | 86 | 95/95 | 183.15 | 90 | 0.0031 |
| 2999000 | 80.9 | 89.02 | 25 | 95/95 | 120.2 | 40 | 0.0032 |
| 3000000 | 89.7 | 89.16 | 68 | 95/95 | 148.45 | 60 | 0.0031 |
