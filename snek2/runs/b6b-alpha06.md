# b6b-alpha06

![b6b-alpha06 progress](b6b-alpha06.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 708000, avg score 54.6, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b6b-alpha06 |
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
| priority_exponent (alpha) | 0.6 |
| priority_signal | td_loss |
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

709 evals so far. Full series in [`b6b-alpha06_evals.json`](b6b-alpha06_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.2 | 0.2 | 0 | 1/95 | -4.808 | 0 | 0.4 |
| 2000 | 0.1 | 0.15 | 0 | 1/95 | -4.902 | 0 | 0.4 |
| ... | | | | | | | |
| 697000 | 49.0 | 52.12 | 3 | 95/95 | 53.97 | 10 | 0.0 |
| 698000 | 37.1 | 51.46 | 2 | 85/95 | 31.821 | 0 | 0.0 |
| 699000 | 39.8 | 44.72 | 1 | 91/95 | 34.393 | 0 | 0.0 |
| 700000 | 60.6 | 45.94 | 5 | 93/95 | 55.09 | 0 | 0.0 |
| 701000 | 65.8 | 50.46 | 19 | 93/95 | 60.197 | 0 | 0.0 |
| 702000 | 53.0 | 51.26 | 3 | 95/95 | 57.97 | 10 | 0.0 |
| 703000 | 27.0 | 49.24 | 1 | 82/95 | 21.735 | 0 | 0.0 |
| 704000 | 80.0 | 57.28 | 64 | 95/95 | 84.66 | 10 | 0.0 |
| 705000 | 59.3 | 57.02 | 3 | 95/95 | 64.127 | 10 | 0.0 |
| 706000 | 37.6 | 51.38 | 1 | 85/95 | 32.291 | 0 | 0.0 |
| 707000 | 56.0 | 51.98 | 2 | 85/95 | 50.504 | 0 | 0.0 |
| 708000 | 54.6 | 57.5 | 3 | 94/95 | 49.05 | 0 | 0.0 |
