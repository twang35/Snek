# smoke

![smoke progress](smoke.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 13000, avg score 16.1, perfect games 0%.

Training was resumed at step 6000, 11000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | smoke |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.99 |
| target_update_period | 8 |
| initial_epsilon | 0.4 |
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| importance_sampling_beta | 0.4 -> 1.0 over 1000000 steps |
| initial_populate_steps | 1000 |
| initialize_with_schmid | False |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.901 | 0 | 0.4 |
| 1000 | 0.6 | 0.6 | 0 | 4/95 | -4.438 | 0 | 0.4 |
| 2000 | 7.6 | 4.1 | 1 | 18/95 | 2.566 | 0 | 0.4 |
| 3000 | 0.7 | 2.97 | 0 | 2/95 | -4.35 | 0 | 0.4 |
| 4000 | 0.4 | 2.32 | 0 | 3/95 | -0.155 | 0 | 0.4 |
| 5000 | 0.3 | 1.92 | 0 | 1/95 | -0.251 | 0 | 0.4 |
| 6000 | 0.4 | 0.4 | 0 | 2/95 | -0.611 | 0 | 0.4 |
| 7000 | 1.6 | 1.6 | 0 | 6/95 | 1.017 | 0 | 0.4 |
| 8000 | 6.0 | 3.8 | 2 | 17/95 | 4.023 | 0 | 0.4 |
| 9000 | 5.5 | 4.37 | 2 | 12/95 | 2.686 | 0 | 0.4 |
| 10000 | 6.4 | 4.88 | 2 | 18/95 | 4.477 | 0 | 0.4 |
| 11000 | 11.4 | 11.4 | 8 | 18/95 | 6.804 | 0 | 0.2 |
| 12000 | 12.7 | 12.7 | 8 | 22/95 | 8.076 | 0 | 0.2 |
| 13000 | 16.1 | 14.4 | 10 | 22/95 | 11.015 | 0 | 0.1 |
