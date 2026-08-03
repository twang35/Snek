# b10c-disc995seed3

![b10c-disc995seed3 progress](b10c-disc995seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 672000, avg score 82.9, perfect games 10%.

## Config

| setting | value |
|---|---|
| policy_name | b10c-disc995seed3 |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.995 |
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
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

673 evals so far. Full series in [`b10c-disc995seed3_evals.json`](b10c-disc995seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.001 | 0 | 0.4 |
| 1000 | 1.9 | 1.9 | 0 | 5/95 | 0.89 | 0 | 0.4 |
| 2000 | 2.7 | 2.3 | 0 | 8/95 | 1.691 | 0 | 0.4 |
| ... | | | | | | | |
| 661000 | 83.0 | 82.52 | 75 | 94/95 | 80.333 | 0 | 0.0 |
| 662000 | 83.1 | 83.28 | 68 | 95/95 | 90.218 | 10 | 0.0 |
| 663000 | 81.2 | 84.08 | 67 | 95/95 | 88.384 | 10 | 0.0 |
| 664000 | 86.4 | 84.24 | 83 | 94/95 | 82.247 | 0 | 0.0 |
| 665000 | 85.6 | 83.86 | 78 | 93/95 | 82.314 | 0 | 0.0 |
| 666000 | 79.4 | 83.14 | 71 | 89/95 | 77.609 | 0 | 0.0 |
| 667000 | 88.6 | 84.24 | 81 | 95/95 | 105.798 | 20 | 0.0 |
| 668000 | 81.5 | 84.3 | 70 | 89/95 | 78.789 | 0 | 0.0 |
| 669000 | 81.4 | 83.3 | 63 | 90/95 | 79.002 | 0 | 0.0 |
| 670000 | 81.0 | 82.38 | 65 | 95/95 | 87.399 | 10 | 0.0 |
| 671000 | 74.9 | 81.48 | 33 | 88/95 | 72.225 | 0 | 0.0 |
| 672000 | 82.9 | 80.34 | 65 | 95/95 | 89.818 | 10 | 0.0 |
