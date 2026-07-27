# smoke

![smoke progress](smoke.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 13000, avg score 16.1, perfect games 0%.

Training was resumed at step 0, 6000, 11000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | smoke |
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

14 evals so far. Full series in [`smoke_evals.json`](smoke_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -5.005 | 0 | 0.4 |
| 2000 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 3000 | 0.1 | 0.03 | 0 | 1/95 | -4.014 | 0 | 0.4 |
| 4000 | 0.4 | 0.12 | 0 | 2/95 | -3.739 | 0 | 0.4 |
| 5000 | 1.5 | 0.4 | 0 | 5/95 | -2.68 | 0 | 0.4 |
| 6000 | 0.8 | 0.56 | 0 | 3/95 | 0.245 | 0 | 0.4 |
| 7000 | 1.1 | 0.78 | 0 | 3/95 | 0.543 | 0 | 0.4 |
| 8000 | 11.1 | 2.98 | 7 | 15/95 | 6.031 | 0 | 0.2 |
| 9000 | 11.3 | 5.16 | 9 | 14/95 | 6.247 | 0 | 0.2 |
| 10000 | 5.0 | 5.86 | 2 | 10/95 | -0.02 | 0 | 0.2 |
| 11000 | 4.8 | 6.66 | 1 | 8/95 | -0.224 | 0 | 0.2 |
| 12000 | 12.7 | 12.7 | 8 | 22/95 | 8.076 | 0 | 0.2 |
| 13000 | 16.1 | 14.4 | 10 | 22/95 | 11.015 | 0 | 0.1 |
