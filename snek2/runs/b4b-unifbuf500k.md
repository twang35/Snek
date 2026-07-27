# b4b-unifbuf500k

![b4b-unifbuf500k progress](b4b-unifbuf500k.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 15000, avg score 10.8, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b4b-unifbuf500k |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.99 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
| initial_epsilon | 0.4 |
| min_epsilon | 0.0 |
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 500000 |
| priority_exponent (alpha) | 0.0 |
| priority_signal | td_error |
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

16 evals so far. Full series in [`b4b-unifbuf500k_evals.json`](b4b-unifbuf500k_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.009 | 0 | 0.4 |
| 1000 | 0.6 | 0.6 | 0 | 2/95 | 0.043 | 0 | 0.4 |
| 2000 | 0.7 | 0.65 | 0 | 3/95 | 0.146 | 0 | 0.4 |
| ... | | | | | | | |
| 4000 | 6.1 | 2.57 | 3 | 12/95 | 1.083 | 0 | 0.4 |
| 5000 | 5.0 | 3.06 | 0 | 10/95 | 0.871 | 0 | 0.4 |
| 6000 | 4.1 | 3.76 | 1 | 7/95 | -0.031 | 0 | 0.4 |
| 7000 | 7.4 | 5.1 | 1 | 11/95 | 2.824 | 0 | 0.4 |
| 8000 | 9.9 | 6.5 | 6 | 15/95 | 4.88 | 0 | 0.4 |
| 9000 | 8.6 | 7.0 | 4 | 13/95 | 3.573 | 0 | 0.4 |
| 10000 | 9.1 | 7.82 | 6 | 13/95 | 4.074 | 0 | 0.4 |
| 11000 | 9.0 | 8.8 | 4 | 15/95 | 3.976 | 0 | 0.4 |
| 12000 | 11.2 | 9.56 | 5 | 14/95 | 6.173 | 0 | 0.2 |
| 13000 | 11.5 | 9.88 | 4 | 15/95 | 6.462 | 0 | 0.2 |
| 14000 | 10.9 | 10.34 | 8 | 14/95 | 5.86 | 0 | 0.2 |
| 15000 | 10.8 | 10.68 | 7 | 15/95 | 5.756 | 0 | 0.2 |
