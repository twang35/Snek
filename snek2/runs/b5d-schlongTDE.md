# b5d-schlongTDE

![b5d-schlongTDE progress](b5d-schlongTDE.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 32000, avg score 48.1, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b5d-schlongTDE |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.99 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
| initial_epsilon | 0.4 |
| min_epsilon | 0.0 |
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.8 |
| priority_signal | td_error |
| importance_sampling_beta | disabled |
| initial_populate_steps | 1000 |
| initialize_with_schmid | False |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |

## Evals

33 evals so far. Full series in [`b5d-schlongTDE_evals.json`](b5d-schlongTDE_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.004 | 0 | 0.4 |
| 1000 | 0.5 | 0.5 | 0 | 1/95 | -0.054 | 0 | 0.4 |
| 2000 | 0.3 | 0.4 | 0 | 1/95 | -1.143 | 0 | 0.4 |
| ... | | | | | | | |
| 21000 | 34.7 | 22.62 | 23 | 47/95 | 30.417 | 0 | 0.05 |
| 22000 | 37.0 | 27.54 | 9 | 61/95 | 33.964 | 0 | 0.05 |
| 23000 | 35.7 | 30.84 | 3 | 61/95 | 32.201 | 0 | 0.05 |
| 24000 | 41.5 | 35.88 | 12 | 60/95 | 38.278 | 0 | 0.05 |
| 25000 | 57.3 | 41.24 | 45 | 69/95 | 53.555 | 0 | 0.01 |
| 26000 | 31.6 | 40.62 | 3 | 63/95 | 29.468 | 0 | 0.01 |
| 27000 | 55.3 | 44.28 | 9 | 77/95 | 51.655 | 0 | 0.01 |
| 28000 | 53.0 | 47.74 | 10 | 75/95 | 49.326 | 0 | 0.01 |
| 29000 | 48.0 | 49.04 | 9 | 71/95 | 44.746 | 0 | 0.01 |
| 30000 | 44.1 | 46.4 | 18 | 75/95 | 41.821 | 0 | 0.01 |
| 31000 | 44.5 | 48.98 | 11 | 75/95 | 41.301 | 0 | 0.01 |
| 32000 | 48.1 | 47.54 | 23 | 70/95 | 44.849 | 0 | 0.01 |
