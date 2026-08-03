# b10a-disc995seed1

![b10a-disc995seed1 progress](b10a-disc995seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2119000, avg score 93.4, perfect games 70%.

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

2120 evals so far. Full series in [`b10a-disc995seed1_evals.json`](b10a-disc995seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.901 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | -2.531 | 0 | 0.4 |
| 2000 | 0.7 | 0.7 | 0 | 2/95 | 0.142 | 0 | 0.4 |
| ... | | | | | | | |
| 2108000 | 93.9 | 92.58 | 87 | 95/95 | 172.335 | 80 | 0.0 |
| 2109000 | 91.4 | 93.08 | 79 | 95/95 | 129.848 | 40 | 0.0 |
| 2110000 | 91.7 | 92.66 | 85 | 95/95 | 130.069 | 40 | 0.0 |
| 2111000 | 93.5 | 92.72 | 91 | 95/95 | 151.883 | 60 | 0.0 |
| 2112000 | 92.0 | 92.5 | 75 | 95/95 | 160.422 | 70 | 0.0 |
| 2113000 | 93.7 | 92.46 | 91 | 95/95 | 152.127 | 60 | 0.0 |
| 2114000 | 92.0 | 92.58 | 81 | 95/95 | 140.05 | 50 | 0.0 |
| 2115000 | 94.4 | 93.12 | 89 | 95/95 | 182.728 | 90 | 0.0 |
| 2116000 | 91.5 | 92.72 | 81 | 95/95 | 129.956 | 40 | 0.0 |
| 2117000 | 91.1 | 92.54 | 74 | 95/95 | 148.706 | 60 | 0.0 |
| 2118000 | 94.5 | 92.7 | 92 | 95/95 | 172.94 | 80 | 0.0 |
| 2119000 | 93.4 | 92.98 | 85 | 95/95 | 161.888 | 70 | 0.0 |
