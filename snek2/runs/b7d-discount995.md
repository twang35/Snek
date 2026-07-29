# b7d-discount995

![b7d-discount995 progress](b7d-discount995.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 4000, avg score 0.1, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b7d-discount995 |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.995 |
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

5 evals so far. Full series in [`b7d-discount995_evals.json`](b7d-discount995_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.902 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 1/95 | -3.698 | 0 | 0.4 |
| 2000 | 0.1 | 0.05 | 0 | 1/95 | -4.49 | 0 | 0.4 |
| 3000 | 0.8 | 0.3 | 0 | 4/95 | -3.355 | 0 | 0.4 |
| 4000 | 0.1 | 0.25 | 0 | 1/95 | -4.907 | 0 | 0.4 |
