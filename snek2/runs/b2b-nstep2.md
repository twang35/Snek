# b2b-nstep2

![b2b-nstep2 progress](b2b-nstep2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 4000, avg score 5.2, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b2b-nstep2 |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.99 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 2 |
| initial_epsilon | 0.4 |
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| importance_sampling_beta | 0.4 -> 1.0 over 1000000 steps |
| initial_populate_steps | 1000 |
| initialize_with_schmid | False |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |

## Evals

5 evals so far. Full series in [`b2b-nstep2_evals.json`](b2b-nstep2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.004 | 0 | 0.4 |
| 1000 | 0.9 | 0.9 | 0 | 2/95 | -1.467 | 0 | 0.4 |
| 2000 | 5.1 | 3.0 | 1 | 8/95 | 0.079 | 0 | 0.4 |
| 3000 | 6.0 | 4.0 | 2 | 14/95 | 0.988 | 0 | 0.4 |
| 4000 | 5.2 | 4.3 | 2 | 11/95 | 0.183 | 0 | 0.4 |
