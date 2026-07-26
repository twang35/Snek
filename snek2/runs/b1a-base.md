# b1a-base

![b1a-base progress](b1a-base.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 175000, avg score 84.1, perfect games 0%.

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

176 evals so far. Full series in [`b1a-base_evals.json`](b1a-base_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.005 | 0 | 0.4 |
| 1000 | 2.9 | 2.9 | 0 | 8/95 | -2.155 | 0 | 0.4 |
| 2000 | 5.1 | 4.0 | 0 | 12/95 | 0.08 | 0 | 0.4 |
| ... | | | | | | | |
| 164000 | 66.7 | 71.12 | 26 | 87/95 | 61.211 | 0 | 0.0 |
| 165000 | 62.0 | 67.38 | 5 | 95/95 | 67.853 | 10 | 0.0 |
| 166000 | 81.0 | 69.58 | 64 | 95/95 | 86.261 | 10 | 0.0 |
| 167000 | 70.3 | 69.1 | 41 | 95/95 | 75.21 | 10 | 0.0 |
| 168000 | 83.4 | 72.68 | 61 | 95/95 | 99.783 | 20 | 0.0 |
| 169000 | 75.5 | 74.44 | 55 | 89/95 | 69.746 | 0 | 0.0 |
| 170000 | 78.2 | 77.68 | 59 | 95/95 | 93.728 | 20 | 0.0 |
| 171000 | 81.5 | 77.78 | 36 | 95/95 | 86.515 | 10 | 0.0 |
| 172000 | 72.8 | 78.28 | 45 | 90/95 | 68.466 | 0 | 0.0 |
| 173000 | 77.3 | 77.06 | 54 | 95/95 | 92.461 | 20 | 0.0 |
| 174000 | 77.3 | 77.42 | 61 | 95/95 | 82.997 | 10 | 0.0 |
| 175000 | 84.1 | 78.6 | 59 | 93/95 | 78.766 | 0 | 0.0 |
