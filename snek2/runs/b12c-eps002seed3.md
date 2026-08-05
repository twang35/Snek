# b12c-eps002seed3

![b12c-eps002seed3 progress](b12c-eps002seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 976000, avg score 57.5, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b12c-eps002seed3 |
| seed | 3 |
| zeroed_observations | none |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.995 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
| initial_epsilon | 0.4 |
| min_epsilon | 0.002 |
| epsilon_schedule | bootstrap on avg_reward [5, 10, 20] then geometric to floor by 80% trailing-30 perfect |
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

977 evals so far. Full series in [`b12c-eps002seed3_evals.json`](b12c-eps002seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.902 | 0 | 0.4 |
| 1000 | 1.0 | 1.0 | 0 | 4/95 | 0.445 | 0 | 0.4 |
| 2000 | 0.9 | 0.95 | 0 | 2/95 | 0.346 | 0 | 0.4 |
| ... | | | | | | | |
| 965000 | 51.8 | 53.06 | 16 | 62/95 | 49.054 | 0 | 0.05 |
| 966000 | 50.6 | 52.46 | 23 | 60/95 | 48.418 | 0 | 0.05 |
| 967000 | 55.1 | 53.1 | 28 | 63/95 | 53.483 | 0 | 0.05 |
| 968000 | 56.9 | 53.18 | 48 | 63/95 | 55.323 | 0 | 0.05 |
| 969000 | 56.8 | 54.24 | 53 | 61/95 | 55.323 | 0 | 0.05 |
| 970000 | 57.2 | 55.32 | 50 | 63/95 | 55.613 | 0 | 0.05 |
| 971000 | 55.3 | 56.26 | 22 | 66/95 | 53.853 | 0 | 0.05 |
| 972000 | 49.7 | 55.18 | 26 | 64/95 | 47.379 | 0 | 0.05 |
| 973000 | 55.8 | 54.96 | 53 | 58/95 | 53.718 | 0 | 0.05 |
| 974000 | 57.3 | 55.06 | 50 | 67/95 | 55.8 | 0 | 0.05 |
| 975000 | 55.9 | 54.8 | 51 | 60/95 | 54.359 | 0 | 0.05 |
| 976000 | 57.5 | 55.24 | 54 | 63/95 | 55.834 | 0 | 0.05 |
