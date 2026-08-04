# b11d-obs30seed4

![b11d-obs30seed4 progress](b11d-obs30seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3594000, avg score 82.6, perfect games 40%.

## Config

| setting | value |
|---|---|
| policy_name | b11d-obs30seed4 |
| seed | 4 |
| zeroed_observations | none |
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

3595 evals so far. Full series in [`b11d-obs30seed4_evals.json`](b11d-obs30seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 0 | 1/95 | -4.503 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | 0.148 | 0 | 0.4 |
| 2000 | 0.7 | 0.7 | 0 | 2/95 | 0.146 | 0 | 0.4 |
| ... | | | | | | | |
| 3583000 | 94.2 | 92.04 | 90 | 95/95 | 172.498 | 80 | 0.0 |
| 3584000 | 93.4 | 92.26 | 86 | 95/95 | 151.445 | 60 | 0.0 |
| 3585000 | 93.7 | 92.24 | 86 | 95/95 | 172.084 | 80 | 0.0 |
| 3586000 | 91.3 | 92.24 | 71 | 95/95 | 149.875 | 60 | 0.0 |
| 3587000 | 94.2 | 93.36 | 90 | 95/95 | 172.235 | 80 | 0.0 |
| 3588000 | 93.8 | 93.28 | 89 | 95/95 | 162.156 | 70 | 0.0 |
| 3589000 | 94.6 | 93.52 | 91 | 95/95 | 182.982 | 90 | 0.0 |
| 3590000 | 93.0 | 93.38 | 85 | 95/95 | 161.422 | 70 | 0.0 |
| 3591000 | 91.1 | 93.34 | 62 | 95/95 | 169.574 | 80 | 0.0 |
| 3592000 | 93.2 | 93.14 | 82 | 95/95 | 161.195 | 70 | 0.0 |
| 3593000 | 95.0 | 93.38 | 95 | 95/95 | 193.392 | 100 | 0.0 |
| 3594000 | 82.6 | 90.98 | 5 | 95/95 | 120.738 | 40 | 0.0 |
