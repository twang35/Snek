# b1a-base

![b1a-base progress](b1a-base.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 108000, avg score 70.3, perfect games 10%.

## Config

| setting | value |
|---|---|
| policy_name | b1a-base |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.99 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
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

109 evals so far. Full series in [`b1a-base_evals.json`](b1a-base_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.005 | 0 | 0.4 |
| 1000 | 2.9 | 2.9 | 0 | 8/95 | -2.155 | 0 | 0.4 |
| 2000 | 5.1 | 4.0 | 0 | 12/95 | 0.08 | 0 | 0.4 |
| ... | | | | | | | |
| 97000 | 68.1 | 70.02 | 42 | 93/95 | 63.446 | 0 | 0.0 |
| 98000 | 70.2 | 70.54 | 13 | 95/95 | 76.651 | 10 | 0.0 |
| 99000 | 70.1 | 69.32 | 52 | 95/95 | 75.407 | 10 | 0.0 |
| 100000 | 77.3 | 68.46 | 42 | 95/95 | 83.672 | 10 | 0.0 |
| 101000 | 75.0 | 72.14 | 32 | 93/95 | 71.1 | 0 | 0.0 |
| 102000 | 75.6 | 73.64 | 53 | 89/95 | 72.047 | 0 | 0.0 |
| 103000 | 65.2 | 72.64 | 19 | 84/95 | 62.217 | 0 | 0.0 |
| 104000 | 60.6 | 70.74 | 25 | 86/95 | 57.063 | 0 | 0.0 |
| 105000 | 65.0 | 68.28 | 46 | 86/95 | 59.852 | 0 | 0.0 |
| 106000 | 78.3 | 68.94 | 64 | 90/95 | 72.627 | 0 | 0.0 |
| 107000 | 65.2 | 66.86 | 40 | 95/95 | 70.997 | 10 | 0.0 |
| 108000 | 70.3 | 67.88 | 26 | 95/95 | 75.912 | 10 | 0.0 |
