# b5a-schlong

![b5a-schlong progress](b5a-schlong.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2046000, avg score 0.1, perfect games 0%.

Training was resumed at step 36000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | b5a-schlong |
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

2047 evals so far. Full series in [`b5a-schlong_evals.json`](b5a-schlong_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.902 | 0 | 0.4 |
| 1000 | 1.4 | 1.4 | 0 | 5/95 | -3.612 | 0 | 0.4 |
| 2000 | 0.0 | 0.7 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| ... | | | | | | | |
| 2035000 | 0.0 | 0.04 | 0 | 0/95 | -5.001 | 0 | 0.0 |
| 2036000 | 0.0 | 0.02 | 0 | 0/95 | -5.002 | 0 | 0.0 |
| 2037000 | 0.0 | 0.02 | 0 | 0/95 | -5.002 | 0 | 0.0 |
| 2038000 | 0.1 | 0.04 | 0 | 1/95 | -4.901 | 0 | 0.0 |
| 2039000 | 0.0 | 0.02 | 0 | 0/95 | -5.002 | 0 | 0.0 |
| 2040000 | 0.0 | 0.02 | 0 | 0/95 | -5.002 | 0 | 0.0 |
| 2041000 | 0.1 | 0.04 | 0 | 1/95 | -4.902 | 0 | 0.0 |
| 2042000 | 0.0 | 0.04 | 0 | 0/95 | -5.001 | 0 | 0.0 |
| 2043000 | 0.0 | 0.02 | 0 | 0/95 | -5.001 | 0 | 0.0 |
| 2044000 | 0.0 | 0.02 | 0 | 0/95 | -5.002 | 0 | 0.0 |
| 2045000 | 0.0 | 0.02 | 0 | 0/95 | -5.001 | 0 | 0.0 |
| 2046000 | 0.1 | 0.02 | 0 | 1/95 | -4.902 | 0 | 0.0 |
