# b7c-a06seed4

![b7c-a06seed4 progress](b7c-a06seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1743000, avg score 0.0, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b7c-a06seed4 |
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

1744 evals so far. Full series in [`b7c-a06seed4_evals.json`](b7c-a06seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 1000 | 0.3 | 0.3 | 0 | 1/95 | -4.75 | 0 | 0.4 |
| 2000 | 1.8 | 1.05 | 0 | 4/95 | -3.259 | 0 | 0.4 |
| ... | | | | | | | |
| 1732000 | 0.0 | 0.0 | 0 | 0/95 | -5.005 | 0 | 0.0 |
| 1733000 | 0.1 | 0.02 | 0 | 1/95 | -4.905 | 0 | 0.0 |
| 1734000 | 0.0 | 0.02 | 0 | 0/95 | -5.006 | 0 | 0.0 |
| 1735000 | 0.0 | 0.02 | 0 | 0/95 | -5.005 | 0 | 0.0 |
| 1736000 | 0.0 | 0.02 | 0 | 0/95 | -5.004 | 0 | 0.0 |
| 1737000 | 0.0 | 0.02 | 0 | 0/95 | -5.007 | 0 | 0.0 |
| 1738000 | 0.1 | 0.02 | 0 | 1/95 | -4.906 | 0 | 0.0 |
| 1739000 | 0.0 | 0.02 | 0 | 0/95 | -5.005 | 0 | 0.0 |
| 1740000 | 0.0 | 0.02 | 0 | 0/95 | -5.006 | 0 | 0.0 |
| 1741000 | 0.0 | 0.02 | 0 | 0/95 | -5.006 | 0 | 0.0 |
| 1742000 | 0.0 | 0.02 | 0 | 0/95 | -5.005 | 0 | 0.0 |
| 1743000 | 0.0 | 0.0 | 0 | 0/95 | -5.006 | 0 | 0.0 |
