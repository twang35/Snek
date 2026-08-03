# b10a-disc995seed1

![b10a-disc995seed1 progress](b10a-disc995seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 854000, avg score 92.4, perfect games 70%.

## Config

| setting | value |
|---|---|
| policy_name | b10a-disc995seed1 |
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

855 evals so far. Full series in [`b10a-disc995seed1_evals.json`](b10a-disc995seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.901 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | -2.531 | 0 | 0.4 |
| 2000 | 0.7 | 0.7 | 0 | 2/95 | 0.142 | 0 | 0.4 |
| ... | | | | | | | |
| 843000 | 92.9 | 90.54 | 81 | 95/95 | 171.407 | 80 | 0.0 |
| 844000 | 92.9 | 91.78 | 74 | 95/95 | 181.02 | 90 | 0.0 |
| 845000 | 93.6 | 91.74 | 81 | 95/95 | 182.034 | 90 | 0.0 |
| 846000 | 93.7 | 91.64 | 87 | 95/95 | 162.21 | 70 | 0.0 |
| 847000 | 95.0 | 93.62 | 95 | 95/95 | 193.421 | 100 | 0.0 |
| 848000 | 94.2 | 93.88 | 91 | 95/95 | 162.572 | 70 | 0.0 |
| 849000 | 92.4 | 93.78 | 85 | 95/95 | 150.954 | 60 | 0.0 |
| 850000 | 83.6 | 91.78 | 51 | 95/95 | 111.758 | 30 | 0.0 |
| 851000 | 91.3 | 91.3 | 61 | 95/95 | 169.849 | 80 | 0.0 |
| 852000 | 87.3 | 89.76 | 55 | 95/95 | 135.82 | 50 | 0.0 |
| 853000 | 94.2 | 89.76 | 87 | 95/95 | 182.692 | 90 | 0.0 |
| 854000 | 92.4 | 89.76 | 76 | 95/95 | 160.446 | 70 | 0.0 |
