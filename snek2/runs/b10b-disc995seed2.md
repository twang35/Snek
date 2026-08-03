# b10b-disc995seed2

![b10b-disc995seed2 progress](b10b-disc995seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 4652000, avg score 93.1, perfect games 50%.

## Config

| setting | value |
|---|---|
| policy_name | b10b-disc995seed2 |
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

4653 evals so far. Full series in [`b10b-disc995seed2_evals.json`](b10b-disc995seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.7 | 0.7 | 0 | 3/95 | -4.301 | 0 | 0.4 |
| 1000 | 1.1 | 1.1 | 0 | 4/95 | 0.083 | 0 | 0.4 |
| 2000 | 7.6 | 4.35 | 0 | 24/95 | 6.516 | 0 | 0.2 |
| ... | | | | | | | |
| 4641000 | 93.1 | 92.44 | 83 | 95/95 | 151.135 | 60 | 0.0 |
| 4642000 | 92.9 | 92.92 | 89 | 95/95 | 151.004 | 60 | 0.0 |
| 4643000 | 93.5 | 93.2 | 89 | 95/95 | 161.547 | 70 | 0.0 |
| 4644000 | 85.2 | 91.44 | 1 | 95/95 | 163.739 | 80 | 0.0 |
| 4645000 | 91.0 | 91.14 | 79 | 95/95 | 159.574 | 70 | 0.0 |
| 4646000 | 94.2 | 91.36 | 91 | 95/95 | 172.674 | 80 | 0.0 |
| 4647000 | 92.6 | 91.3 | 89 | 95/95 | 130.775 | 40 | 0.0 |
| 4648000 | 92.4 | 91.08 | 83 | 95/95 | 150.515 | 60 | 0.0 |
| 4649000 | 93.9 | 92.82 | 89 | 95/95 | 162.36 | 70 | 0.0 |
| 4650000 | 94.4 | 93.5 | 91 | 95/95 | 172.87 | 80 | 0.0 |
| 4651000 | 91.6 | 92.98 | 74 | 95/95 | 149.202 | 60 | 0.0 |
| 4652000 | 93.1 | 93.08 | 83 | 95/95 | 141.2 | 50 | 0.0 |
