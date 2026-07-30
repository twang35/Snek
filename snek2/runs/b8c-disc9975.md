# b8c-disc9975

![b8c-disc9975 progress](b8c-disc9975.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1754000, avg score 12.1, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b8c-disc9975 |
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

1755 evals so far. Full series in [`b8c-disc9975_evals.json`](b8c-disc9975_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.004 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 2000 | 0.0 | 0.0 | 0 | 0/95 | -4.599 | 0 | 0.4 |
| ... | | | | | | | |
| 1743000 | 8.4 | 9.16 | 3 | 13/95 | 3.384 | 0 | 0.0 |
| 1744000 | 8.7 | 8.66 | 3 | 14/95 | 3.686 | 0 | 0.0 |
| 1745000 | 7.6 | 8.4 | 1 | 12/95 | 2.587 | 0 | 0.0 |
| 1746000 | 10.2 | 8.76 | 3 | 16/95 | 5.183 | 0 | 0.0 |
| 1747000 | 9.6 | 8.9 | 1 | 16/95 | 4.582 | 0 | 0.0 |
| 1748000 | 12.4 | 9.7 | 3 | 19/95 | 7.378 | 0 | 0.0 |
| 1749000 | 8.4 | 9.64 | 3 | 12/95 | 3.384 | 0 | 0.0 |
| 1750000 | 9.0 | 9.92 | 3 | 13/95 | 3.985 | 0 | 0.0 |
| 1751000 | 11.0 | 10.08 | 5 | 17/95 | 5.98 | 0 | 0.0 |
| 1752000 | 12.2 | 10.6 | 8 | 16/95 | 7.179 | 0 | 0.0 |
| 1753000 | 11.0 | 10.32 | 6 | 17/95 | 5.979 | 0 | 0.0 |
| 1754000 | 12.1 | 11.06 | 4 | 21/95 | 7.076 | 0 | 0.0 |
