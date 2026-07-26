# b1c-nstep3

![b1c-nstep3 progress](b1c-nstep3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 178000, avg score 44.0, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b1c-nstep3 |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.99 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 3 |
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

179 evals so far. Full series in [`b1c-nstep3_evals.json`](b1c-nstep3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.2 | 0.2 | 0 | 1/95 | -4.808 | 0 | 0.4 |
| 1000 | 1.8 | 1.8 | 0 | 5/95 | -1.0 | 0 | 0.4 |
| 2000 | 0.0 | 0.9 | 0 | 0/95 | -5.005 | 0 | 0.4 |
| ... | | | | | | | |
| 167000 | 39.2 | 42.1 | 22 | 72/95 | 34.031 | 0 | 0.001 |
| 168000 | 41.7 | 41.12 | 15 | 76/95 | 36.472 | 0 | 0.001 |
| 169000 | 41.9 | 41.62 | 9 | 78/95 | 37.09 | 0 | 0.001 |
| 170000 | 47.4 | 40.9 | 16 | 85/95 | 42.536 | 0 | 0.001 |
| 171000 | 38.4 | 41.72 | 8 | 78/95 | 33.59 | 0 | 0.001 |
| 172000 | 53.4 | 44.56 | 8 | 80/95 | 47.966 | 0 | 0.001 |
| 173000 | 56.7 | 47.56 | 18 | 91/95 | 51.282 | 0 | 0.001 |
| 174000 | 50.2 | 49.22 | 30 | 73/95 | 44.835 | 0 | 0.001 |
| 175000 | 33.6 | 46.46 | 10 | 79/95 | 28.405 | 0 | 0.001 |
| 176000 | 48.2 | 48.42 | 26 | 76/95 | 42.837 | 0 | 0.001 |
| 177000 | 54.9 | 48.72 | 30 | 80/95 | 49.564 | 0 | 0.001 |
| 178000 | 44.0 | 46.18 | 20 | 66/95 | 38.743 | 0 | 0.001 |
