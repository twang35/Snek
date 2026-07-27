# b4c-schlongper

![b4c-schlongper progress](b4c-schlongper.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 16000, avg score 5.5, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b4c-schlongper |
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

17 evals so far. Full series in [`b4c-schlongper_evals.json`](b4c-schlongper_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 4/95 | -4.316 | 0 | 0.4 |
| 2000 | 0.5 | 0.6 | 0 | 4/95 | -4.55 | 0 | 0.4 |
| ... | | | | | | | |
| 5000 | 0.4 | 0.48 | 0 | 1/95 | -0.154 | 0 | 0.4 |
| 6000 | 0.3 | 0.4 | 0 | 1/95 | -0.704 | 0 | 0.4 |
| 7000 | 0.8 | 0.46 | 0 | 2/95 | 0.246 | 0 | 0.4 |
| 8000 | 3.6 | 1.04 | 0 | 7/95 | 0.366 | 0 | 0.4 |
| 9000 | 1.3 | 1.28 | 0 | 4/95 | -3.703 | 0 | 0.4 |
| 10000 | 2.8 | 1.76 | 0 | 9/95 | -2.205 | 0 | 0.4 |
| 11000 | 1.7 | 2.04 | 0 | 7/95 | -3.303 | 0 | 0.4 |
| 12000 | 2.7 | 2.42 | 0 | 8/95 | -2.305 | 0 | 0.4 |
| 13000 | 2.9 | 2.28 | 0 | 9/95 | -2.105 | 0 | 0.4 |
| 14000 | 2.0 | 2.42 | 0 | 8/95 | -3.004 | 0 | 0.4 |
| 15000 | 3.1 | 2.48 | 0 | 9/95 | -1.905 | 0 | 0.4 |
| 16000 | 5.5 | 3.24 | 0 | 10/95 | 0.491 | 0 | 0.4 |
