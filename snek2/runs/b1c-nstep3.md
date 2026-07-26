# b1c-nstep3

![b1c-nstep3 progress](b1c-nstep3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 306000, avg score 53.2, perfect games 0%.

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

307 evals so far. Full series in [`b1c-nstep3_evals.json`](b1c-nstep3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.2 | 0.2 | 0 | 1/95 | -4.808 | 0 | 0.4 |
| 1000 | 1.8 | 1.8 | 0 | 5/95 | -1.0 | 0 | 0.4 |
| 2000 | 0.0 | 0.9 | 0 | 0/95 | -5.005 | 0 | 0.4 |
| ... | | | | | | | |
| 295000 | 51.3 | 49.2 | 13 | 80/95 | 46.46 | 0 | 0.001 |
| 296000 | 57.8 | 51.18 | 27 | 86/95 | 52.943 | 0 | 0.001 |
| 297000 | 49.8 | 51.16 | 35 | 70/95 | 44.55 | 0 | 0.001 |
| 298000 | 51.7 | 50.72 | 32 | 82/95 | 46.436 | 0 | 0.001 |
| 299000 | 43.4 | 50.8 | 14 | 68/95 | 39.091 | 0 | 0.001 |
| 300000 | 44.6 | 49.46 | 20 | 64/95 | 39.392 | 0 | 0.001 |
| 301000 | 46.3 | 47.16 | 32 | 60/95 | 41.11 | 0 | 0.001 |
| 302000 | 49.0 | 47.0 | 20 | 72/95 | 44.209 | 0 | 0.001 |
| 303000 | 52.3 | 47.12 | 39 | 70/95 | 47.021 | 0 | 0.001 |
| 304000 | 48.2 | 48.08 | 20 | 74/95 | 42.98 | 0 | 0.001 |
| 305000 | 44.8 | 48.12 | 25 | 62/95 | 40.04 | 0 | 0.001 |
| 306000 | 53.2 | 49.5 | 31 | 62/95 | 47.939 | 0 | 0.001 |
