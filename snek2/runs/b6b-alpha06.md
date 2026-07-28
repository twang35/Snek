# b6b-alpha06

![b6b-alpha06 progress](b6b-alpha06.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1803000, avg score 46.1, perfect games 10%.

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

1804 evals so far. Full series in [`b6b-alpha06_evals.json`](b6b-alpha06_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.2 | 0.2 | 0 | 1/95 | -4.808 | 0 | 0.4 |
| 2000 | 0.1 | 0.15 | 0 | 1/95 | -4.902 | 0 | 0.4 |
| ... | | | | | | | |
| 1792000 | 37.0 | 30.9 | 2 | 76/95 | 31.32 | 0 | 0.0 |
| 1793000 | 12.7 | 27.28 | 0 | 62/95 | 7.544 | 0 | 0.0 |
| 1794000 | 7.4 | 27.64 | 1 | 26/95 | 2.296 | 0 | 0.0 |
| 1795000 | 5.5 | 19.68 | 0 | 17/95 | 0.427 | 0 | 0.0 |
| 1796000 | 7.1 | 13.94 | 0 | 26/95 | 2.002 | 0 | 0.0 |
| 1797000 | 27.3 | 12.0 | 1 | 87/95 | 21.845 | 0 | 0.0 |
| 1798000 | 31.6 | 15.78 | 0 | 91/95 | 26.095 | 0 | 0.0 |
| 1799000 | 34.6 | 21.22 | 0 | 94/95 | 29.124 | 0 | 0.0 |
| 1800000 | 42.5 | 28.62 | 0 | 85/95 | 36.917 | 0 | 0.0 |
| 1801000 | 32.7 | 33.74 | 0 | 95/95 | 48.089 | 20 | 0.0 |
| 1802000 | 11.6 | 30.6 | 1 | 79/95 | 6.438 | 0 | 0.0 |
| 1803000 | 46.1 | 33.5 | 1 | 95/95 | 50.862 | 10 | 0.0 |
