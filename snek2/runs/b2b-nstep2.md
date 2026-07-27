# b2b-nstep2

![b2b-nstep2 progress](b2b-nstep2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 580000, avg score 38.3, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b2b-nstep2 |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.99 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 2 |
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

581 evals so far. Full series in [`b2b-nstep2_evals.json`](b2b-nstep2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.004 | 0 | 0.4 |
| 1000 | 0.9 | 0.9 | 0 | 2/95 | -1.467 | 0 | 0.4 |
| 2000 | 5.1 | 3.0 | 1 | 8/95 | 0.079 | 0 | 0.4 |
| ... | | | | | | | |
| 569000 | 45.0 | 36.42 | 38 | 55/95 | 39.83 | 0 | 0.001 |
| 570000 | 38.3 | 37.88 | 20 | 48/95 | 33.142 | 0 | 0.001 |
| 571000 | 34.6 | 37.68 | 5 | 54/95 | 29.901 | 0 | 0.001 |
| 572000 | 36.8 | 37.92 | 24 | 53/95 | 31.658 | 0 | 0.001 |
| 573000 | 39.2 | 38.78 | 16 | 52/95 | 34.03 | 0 | 0.001 |
| 574000 | 34.0 | 36.58 | 16 | 42/95 | 29.315 | 0 | 0.001 |
| 575000 | 34.3 | 35.78 | 20 | 46/95 | 29.609 | 0 | 0.001 |
| 576000 | 43.2 | 37.5 | 22 | 66/95 | 38.025 | 0 | 0.001 |
| 577000 | 38.0 | 37.74 | 18 | 48/95 | 32.817 | 0 | 0.001 |
| 578000 | 42.2 | 38.34 | 18 | 53/95 | 37.009 | 0 | 0.001 |
| 579000 | 36.9 | 38.92 | 12 | 49/95 | 31.747 | 0 | 0.001 |
| 580000 | 38.3 | 39.72 | 20 | 48/95 | 33.558 | 0 | 0.001 |
