# b8f-disc9975seed2

![b8f-disc9975seed2 progress](b8f-disc9975seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1805000, avg score 76.4, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b8f-disc9975seed2 |
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
| initialize_with_schmid | False |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |
| perfect_game_wait_ms | 500 |

## Evals

1806 evals so far. Full series in [`b8f-disc9975seed2_evals.json`](b8f-disc9975seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.3 | 0.3 | 0 | 1/95 | -4.703 | 0 | 0.4 |
| 1000 | 11.3 | 11.3 | 0 | 17/95 | 6.262 | 0 | 0.2 |
| 2000 | 0.1 | 5.7 | 0 | 1/95 | -4.906 | 0 | 0.2 |
| ... | | | | | | | |
| 1794000 | 82.7 | 79.14 | 39 | 95/95 | 97.654 | 20 | 0.0 |
| 1795000 | 76.3 | 78.08 | 40 | 95/95 | 81.07 | 10 | 0.0 |
| 1796000 | 82.3 | 79.02 | 64 | 95/95 | 118.211 | 40 | 0.0 |
| 1797000 | 69.3 | 78.16 | 32 | 95/95 | 74.047 | 10 | 0.0 |
| 1798000 | 85.7 | 79.26 | 60 | 95/95 | 121.621 | 40 | 0.0 |
| 1799000 | 56.4 | 74.0 | 25 | 90/95 | 50.892 | 0 | 0.0 |
| 1800000 | 78.4 | 74.42 | 47 | 95/95 | 114.405 | 40 | 0.0 |
| 1801000 | 76.0 | 73.16 | 13 | 95/95 | 102.032 | 30 | 0.0 |
| 1802000 | 90.1 | 77.32 | 74 | 95/95 | 136.204 | 50 | 0.0 |
| 1803000 | 82.0 | 76.58 | 57 | 95/95 | 107.51 | 30 | 0.0 |
| 1804000 | 80.1 | 81.32 | 62 | 95/95 | 95.134 | 20 | 0.0 |
| 1805000 | 76.4 | 80.92 | 42 | 94/95 | 70.669 | 0 | 0.0 |
