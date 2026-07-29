# b7d-discount995

![b7d-discount995 progress](b7d-discount995.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 647000, avg score 71.6, perfect games 10%.

## Config

| setting | value |
|---|---|
| policy_name | b7d-discount995 |
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
| initialize_with_schmid | False |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |

## Evals

648 evals so far. Full series in [`b7d-discount995_evals.json`](b7d-discount995_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.902 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 1/95 | -3.698 | 0 | 0.4 |
| 2000 | 0.1 | 0.05 | 0 | 1/95 | -4.49 | 0 | 0.4 |
| ... | | | | | | | |
| 636000 | 69.8 | 60.48 | 14 | 95/95 | 74.497 | 10 | 0.0 |
| 637000 | 53.2 | 62.04 | 7 | 95/95 | 58.073 | 10 | 0.0 |
| 638000 | 77.0 | 65.74 | 57 | 92/95 | 71.123 | 0 | 0.0 |
| 639000 | 62.7 | 64.32 | 19 | 88/95 | 57.069 | 0 | 0.0 |
| 640000 | 68.7 | 66.28 | 43 | 86/95 | 63.115 | 0 | 0.0 |
| 641000 | 67.2 | 65.76 | 30 | 91/95 | 61.454 | 0 | 0.0 |
| 642000 | 59.2 | 66.96 | 12 | 90/95 | 53.56 | 0 | 0.0 |
| 643000 | 64.2 | 64.4 | 12 | 95/95 | 68.882 | 10 | 0.0 |
| 644000 | 70.5 | 65.96 | 9 | 95/95 | 96.045 | 30 | 0.0 |
| 645000 | 72.0 | 66.62 | 9 | 92/95 | 66.182 | 0 | 0.0 |
| 646000 | 62.4 | 65.66 | 13 | 95/95 | 77.359 | 20 | 0.0 |
| 647000 | 71.6 | 68.14 | 13 | 95/95 | 76.356 | 10 | 0.0 |
