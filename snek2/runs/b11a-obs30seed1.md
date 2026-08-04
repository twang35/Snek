# b11a-obs30seed1

![b11a-obs30seed1 progress](b11a-obs30seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3185000, avg score 82.4, perfect games 10%.

## Config

| setting | value |
|---|---|
| policy_name | b11a-obs30seed1 |
| seed | 1 |
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

3186 evals so far. Full series in [`b11a-obs30seed1_evals.json`](b11a-obs30seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -0.549 | 0 | 0.4 |
| 2000 | 0.0 | 0.0 | 0 | 0/95 | -0.549 | 0 | 0.4 |
| ... | | | | | | | |
| 3174000 | 94.6 | 88.84 | 92 | 95/95 | 172.343 | 80 | 0.0 |
| 3175000 | 93.0 | 90.4 | 79 | 95/95 | 160.92 | 70 | 0.0 |
| 3176000 | 92.0 | 91.16 | 82 | 95/95 | 129.901 | 40 | 0.0 |
| 3177000 | 93.8 | 92.08 | 92 | 95/95 | 151.684 | 60 | 0.0 |
| 3178000 | 93.4 | 93.36 | 90 | 95/95 | 141.619 | 50 | 0.0 |
| 3179000 | 89.8 | 92.4 | 76 | 95/95 | 116.829 | 30 | 0.0 |
| 3180000 | 88.1 | 91.42 | 78 | 95/95 | 116.529 | 30 | 0.0 |
| 3181000 | 91.4 | 91.3 | 76 | 95/95 | 149.322 | 60 | 0.0 |
| 3182000 | 85.4 | 89.62 | 74 | 95/95 | 102.904 | 20 | 0.0 |
| 3183000 | 87.8 | 88.5 | 56 | 95/95 | 125.746 | 40 | 0.0 |
| 3184000 | 90.5 | 88.64 | 82 | 95/95 | 128.694 | 40 | 0.0 |
| 3185000 | 82.4 | 87.5 | 23 | 95/95 | 89.936 | 10 | 0.0 |
