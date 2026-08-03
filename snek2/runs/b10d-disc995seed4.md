# b10d-disc995seed4

![b10d-disc995seed4 progress](b10d-disc995seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 826000, avg score 86.5, perfect games 70%.

## Config

| setting | value |
|---|---|
| policy_name | b10d-disc995seed4 |
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

827 evals so far. Full series in [`b10d-disc995seed4_evals.json`](b10d-disc995seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.004 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 2/95 | -0.152 | 0 | 0.4 |
| 2000 | 0.0 | 0.2 | 0 | 0/95 | -0.548 | 0 | 0.4 |
| ... | | | | | | | |
| 815000 | 88.7 | 90.3 | 62 | 95/95 | 146.355 | 60 | 0.0 |
| 816000 | 94.3 | 92.64 | 91 | 95/95 | 172.714 | 80 | 0.0 |
| 817000 | 93.0 | 92.52 | 81 | 95/95 | 161.328 | 70 | 0.0 |
| 818000 | 94.8 | 92.9 | 93 | 95/95 | 183.206 | 90 | 0.0 |
| 819000 | 92.2 | 92.6 | 77 | 95/95 | 160.612 | 70 | 0.0 |
| 820000 | 92.6 | 93.38 | 83 | 95/95 | 171.05 | 80 | 0.0 |
| 821000 | 86.7 | 91.86 | 47 | 95/95 | 145.282 | 60 | 0.0 |
| 822000 | 94.3 | 92.12 | 91 | 95/95 | 172.778 | 80 | 0.0 |
| 823000 | 85.1 | 90.18 | 29 | 95/95 | 143.736 | 60 | 0.0 |
| 824000 | 93.1 | 90.36 | 88 | 95/95 | 161.554 | 70 | 0.0 |
| 825000 | 92.9 | 90.42 | 85 | 95/95 | 150.979 | 60 | 0.0 |
| 826000 | 86.5 | 90.38 | 21 | 95/95 | 154.541 | 70 | 0.0 |
