# b2a-base2

![b2a-base2 progress](b2a-base2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 24000, avg score 54.8, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b2a-base2 |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.99 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
| initial_epsilon | 0.4 |
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| importance_sampling_beta | 0.4 -> 1.0 over 1000000 steps |
| initial_populate_steps | 1000 |
| initialize_with_schmid | False |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |

## Evals

25 evals so far. Full series in [`b2a-base2_evals.json`](b2a-base2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 1/95 | -0.153 | 0 | 0.4 |
| 2000 | 1.1 | 0.75 | 0 | 4/95 | 0.543 | 0 | 0.4 |
| ... | | | | | | | |
| 13000 | 14.9 | 14.0 | 10 | 23/95 | 9.866 | 0 | 0.1 |
| 14000 | 16.2 | 14.28 | 10 | 22/95 | 11.166 | 0 | 0.1 |
| 15000 | 16.8 | 15.26 | 8 | 24/95 | 11.767 | 0 | 0.1 |
| 16000 | 18.5 | 16.3 | 13 | 27/95 | 13.887 | 0 | 0.1 |
| 17000 | 18.7 | 17.02 | 5 | 28/95 | 14.077 | 0 | 0.1 |
| 18000 | 19.8 | 18.0 | 10 | 27/95 | 15.143 | 0 | 0.1 |
| 19000 | 26.1 | 19.98 | 11 | 35/95 | 20.985 | 0 | 0.05 |
| 20000 | 40.2 | 24.66 | 28 | 50/95 | 34.901 | 0 | 0.05 |
| 21000 | 42.2 | 29.4 | 9 | 68/95 | 37.215 | 0 | 0.05 |
| 22000 | 52.9 | 36.24 | 7 | 84/95 | 48.231 | 0 | 0.01 |
| 23000 | 45.2 | 41.32 | 2 | 76/95 | 40.248 | 0 | 0.01 |
| 24000 | 54.8 | 47.06 | 18 | 80/95 | 49.385 | 0 | 0.01 |
