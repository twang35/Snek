# b9b-disc9975b

![b9b-disc9975b progress](b9b-disc9975b.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 10466000, avg score 0.7, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b9b-disc9975b |
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

10467 evals so far. Full series in [`b9b-disc9975b_evals.json`](b9b-disc9975b_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.903 | 0 | 0.4 |
| 1000 | 0.2 | 0.2 | 0 | 1/95 | -4.803 | 0 | 0.4 |
| 2000 | 0.2 | 0.2 | 0 | 2/95 | -4.802 | 0 | 0.4 |
| ... | | | | | | | |
| 10455000 | 0.4 | 0.48 | 0 | 1/95 | -4.601 | 0 | 0.001 |
| 10456000 | 0.7 | 0.48 | 0 | 3/95 | -4.301 | 0 | 0.001 |
| 10457000 | 0.4 | 0.44 | 0 | 1/95 | -4.601 | 0 | 0.001 |
| 10458000 | 0.1 | 0.38 | 0 | 1/95 | -4.9 | 0 | 0.001 |
| 10459000 | 0.2 | 0.36 | 0 | 1/95 | -4.801 | 0 | 0.001 |
| 10460000 | 0.4 | 0.36 | 0 | 1/95 | -4.601 | 0 | 0.001 |
| 10461000 | 0.4 | 0.3 | 0 | 2/95 | -4.6 | 0 | 0.001 |
| 10462000 | 0.3 | 0.28 | 0 | 1/95 | -4.701 | 0 | 0.001 |
| 10463000 | 0.6 | 0.38 | 0 | 2/95 | -4.401 | 0 | 0.001 |
| 10464000 | 0.3 | 0.4 | 0 | 1/95 | -4.701 | 0 | 0.001 |
| 10465000 | 0.1 | 0.34 | 0 | 1/95 | -4.9 | 0 | 0.001 |
| 10466000 | 0.7 | 0.4 | 0 | 3/95 | -4.301 | 0 | 0.001 |
