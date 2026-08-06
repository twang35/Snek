# b14d-disc9975seed4

![b14d-disc9975seed4 progress](b14d-disc9975seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 4461000, avg score 94.6, perfect games 80%.

## Config

| setting | value |
|---|---|
| policy_name | b14d-disc9975seed4 |
| seed | 4 |
| zeroed_observations | none |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.9975 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
| initial_epsilon | 0.4 |
| min_epsilon | 0.002 |
| epsilon_schedule | bootstrap on avg_reward [2, 5, 10, 15, 20] then geometric to floor by 80% trailing-30 perfect |
| guided_fraction | 0.8 |
| exploration_shield | 80% of refinement-phase episodes draw the epsilon move from non-fatal actions; greedy moves never shielded |
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | td_loss |
| importance_sampling_beta | disabled |
| max_steps | 5000000 |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

4462 evals so far. Full series in [`b14d-disc9975seed4_evals.json`](b14d-disc9975seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 0 | 1/95 | -4.503 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | 0.147 | 0 | 0.4 |
| 2000 | 0.7 | 0.7 | 0 | 2/95 | 0.146 | 0 | 0.4 |
| ... | | | | | | | |
| 4450000 | 94.4 | 94.66 | 92 | 95/95 | 171.892 | 80 | 0.0022 |
| 4451000 | 94.3 | 94.56 | 92 | 95/95 | 161.432 | 70 | 0.0021 |
| 4452000 | 93.8 | 94.32 | 85 | 95/95 | 171.337 | 80 | 0.0022 |
| 4453000 | 94.3 | 94.26 | 90 | 95/95 | 171.873 | 80 | 0.0021 |
| 4454000 | 92.3 | 93.82 | 73 | 95/95 | 169.913 | 80 | 0.0021 |
| 4455000 | 93.9 | 93.72 | 91 | 95/95 | 150.361 | 60 | 0.0021 |
| 4456000 | 95.0 | 93.86 | 95 | 95/95 | 193.349 | 100 | 0.0021 |
| 4457000 | 94.5 | 94.0 | 93 | 95/95 | 161.636 | 70 | 0.0021 |
| 4458000 | 94.7 | 94.08 | 92 | 95/95 | 182.708 | 90 | 0.0021 |
| 4459000 | 94.7 | 94.56 | 92 | 95/95 | 182.55 | 90 | 0.0021 |
| 4460000 | 94.7 | 94.72 | 93 | 95/95 | 172.248 | 80 | 0.0021 |
| 4461000 | 94.6 | 94.64 | 93 | 95/95 | 172.152 | 80 | 0.0021 |
