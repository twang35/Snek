# b20ac-fc93x93seed3

![b20ac-fc93x93seed3 progress](b20ac-fc93x93seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 92.6, perfect games 40%.

## Config

| setting | value |
|---|---|
| policy_name | b20ac-fc93x93seed3 |
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
| fc_layer_params | (93, 93) |
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

3001 evals so far. Full series in [`b20ac-fc93x93seed3_evals.json`](b20ac-fc93x93seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.1 | 0.1 | 0 | 1/95 | -4.0 | 0 | 0.4 |
| 2000 | 0.0 | 0.05 | 0 | 0/95 | -0.5 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.1 | 93.36 | 93 | 95/95 | 142.9 | 50 | 0.0048 |
| 2990000 | 93.1 | 93.34 | 90 | 95/95 | 122.45 | 30 | 0.0049 |
| 2991000 | 93.5 | 93.48 | 91 | 95/95 | 132.35 | 40 | 0.005 |
| 2992000 | 94.0 | 93.7 | 91 | 95/95 | 162.7 | 70 | 0.0048 |
| 2993000 | 92.7 | 93.48 | 84 | 95/95 | 141.95 | 50 | 0.0048 |
| 2994000 | 92.8 | 93.22 | 84 | 95/95 | 131.65 | 40 | 0.0047 |
| 2995000 | 94.1 | 93.42 | 91 | 95/95 | 152.4 | 60 | 0.0047 |
| 2996000 | 93.3 | 93.38 | 88 | 95/95 | 152.5 | 60 | 0.0047 |
| 2997000 | 93.8 | 93.34 | 91 | 95/95 | 153.0 | 60 | 0.0046 |
| 2998000 | 86.2 | 92.04 | 24 | 95/95 | 115.1 | 30 | 0.0046 |
| 2999000 | 90.1 | 91.5 | 59 | 95/95 | 119.0 | 30 | 0.0046 |
| 3000000 | 92.6 | 91.2 | 87 | 95/95 | 130.55 | 40 | 0.0045 |
