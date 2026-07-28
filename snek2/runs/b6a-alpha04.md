# b6a-alpha04

![b6a-alpha04 progress](b6a-alpha04.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 10000, avg score 4.9, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b6a-alpha04 |
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
| priority_exponent (alpha) | 0.4 |
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

11 evals so far. Full series in [`b6a-alpha04_evals.json`](b6a-alpha04_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -5.053 | 0 | 0.4 |
| 2000 | 0.0 | 0.0 | 0 | 0/95 | -1.001 | 0 | 0.4 |
| 3000 | 0.0 | 0.0 | 0 | 0/95 | -0.553 | 0 | 0.4 |
| 4000 | 0.1 | 0.03 | 0 | 1/95 | -0.456 | 0 | 0.4 |
| 5000 | 2.0 | 0.42 | 0 | 6/95 | 1.416 | 0 | 0.4 |
| 6000 | 1.6 | 0.74 | 0 | 5/95 | 1.011 | 0 | 0.4 |
| 7000 | 1.6 | 1.06 | 0 | 3/95 | -1.676 | 0 | 0.4 |
| 8000 | 1.0 | 1.26 | 0 | 2/95 | -2.254 | 0 | 0.4 |
| 9000 | 2.5 | 1.74 | 0 | 6/95 | -2.123 | 0 | 0.4 |
| 10000 | 4.9 | 2.32 | 2 | 9/95 | -0.165 | 0 | 0.4 |
