# b9a-disc9975a

![b9a-disc9975a progress](b9a-disc9975a.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3676000, avg score 46.9, perfect games 30%.

## Config

| setting | value |
|---|---|
| policy_name | b9a-disc9975a |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.9975 |
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

3677 evals so far. Full series in [`b9a-disc9975a_evals.json`](b9a-disc9975a_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.1 | 0.1 | 0 | 1/95 | -4.903 | 0 | 0.4 |
| 2000 | 0.5 | 0.3 | 0 | 2/95 | -4.55 | 0 | 0.4 |
| ... | | | | | | | |
| 3665000 | 77.9 | 74.38 | 24 | 95/95 | 82.496 | 10 | 0.0 |
| 3666000 | 85.0 | 78.16 | 21 | 95/95 | 162.117 | 80 | 0.0 |
| 3667000 | 77.3 | 76.88 | 27 | 95/95 | 123.627 | 50 | 0.0 |
| 3668000 | 88.9 | 81.32 | 59 | 95/95 | 145.605 | 60 | 0.0 |
| 3669000 | 82.0 | 82.22 | 22 | 95/95 | 107.458 | 30 | 0.0 |
| 3670000 | 77.1 | 82.06 | 12 | 95/95 | 123.424 | 50 | 0.0 |
| 3671000 | 74.2 | 79.9 | 19 | 95/95 | 110.134 | 40 | 0.0 |
| 3672000 | 73.9 | 79.22 | 42 | 95/95 | 78.584 | 10 | 0.0 |
| 3673000 | 86.4 | 78.72 | 58 | 95/95 | 122.195 | 40 | 0.0 |
| 3674000 | 68.3 | 75.98 | 1 | 95/95 | 83.463 | 20 | 0.0 |
| 3675000 | 65.9 | 73.74 | 4 | 95/95 | 70.649 | 10 | 0.0 |
| 3676000 | 46.9 | 68.28 | 1 | 95/95 | 72.663 | 30 | 0.0 |
