# b5b-schlong2

![b5b-schlong2 progress](b5b-schlong2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1922000, avg score 0.0, perfect games 0%.

Training was resumed at step 46000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | b5b-schlong2 |
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

1923 evals so far. Full series in [`b5b-schlong2_evals.json`](b5b-schlong2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.004 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -5.008 | 0 | 0.4 |
| 2000 | 0.0 | 0.0 | 0 | 0/95 | -5.007 | 0 | 0.4 |
| ... | | | | | | | |
| 1911000 | 0.0 | 0.04 | 0 | 0/95 | -5.002 | 0 | 0.0 |
| 1912000 | 0.0 | 0.04 | 0 | 0/95 | -5.002 | 0 | 0.0 |
| 1913000 | 0.1 | 0.04 | 0 | 1/95 | -4.901 | 0 | 0.0 |
| 1914000 | 0.0 | 0.02 | 0 | 0/95 | -5.002 | 0 | 0.0 |
| 1915000 | 0.1 | 0.04 | 0 | 1/95 | -4.902 | 0 | 0.0 |
| 1916000 | 0.0 | 0.04 | 0 | 0/95 | -5.002 | 0 | 0.0 |
| 1917000 | 0.0 | 0.04 | 0 | 0/95 | -5.001 | 0 | 0.0 |
| 1918000 | 0.1 | 0.04 | 0 | 1/95 | -4.901 | 0 | 0.0 |
| 1919000 | 0.1 | 0.06 | 0 | 1/95 | -4.901 | 0 | 0.0 |
| 1920000 | 0.0 | 0.04 | 0 | 0/95 | -5.002 | 0 | 0.0 |
| 1921000 | 0.0 | 0.04 | 0 | 0/95 | -5.001 | 0 | 0.0 |
| 1922000 | 0.0 | 0.04 | 0 | 0/95 | -5.001 | 0 | 0.0 |
