# b14a-disc9975seed1

![b14a-disc9975seed1 progress](b14a-disc9975seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1348000, avg score 92.0, perfect games 30%.

## Config

| setting | value |
|---|---|
| policy_name | b14a-disc9975seed1 |
| seed | 1 |
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

1349 evals so far. Full series in [`b14a-disc9975seed1_evals.json`](b14a-disc9975seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -0.55 | 0 | 0.4 |
| 2000 | 0.0 | 0.0 | 0 | 0/95 | -3.696 | 0 | 0.4 |
| ... | | | | | | | |
| 1337000 | 90.6 | 91.24 | 82 | 95/95 | 97.327 | 10 | 0.0054 |
| 1338000 | 93.8 | 91.46 | 91 | 95/95 | 140.963 | 50 | 0.0054 |
| 1339000 | 85.2 | 90.22 | 50 | 95/95 | 113.369 | 30 | 0.0054 |
| 1340000 | 90.1 | 90.16 | 82 | 95/95 | 118.113 | 30 | 0.0054 |
| 1341000 | 90.8 | 90.1 | 60 | 95/95 | 138.005 | 50 | 0.0054 |
| 1342000 | 92.7 | 90.52 | 90 | 95/95 | 131.186 | 40 | 0.0055 |
| 1343000 | 94.3 | 90.62 | 92 | 95/95 | 162.321 | 70 | 0.0055 |
| 1344000 | 92.8 | 92.14 | 83 | 95/95 | 160.375 | 70 | 0.0054 |
| 1345000 | 92.4 | 92.6 | 84 | 95/95 | 140.372 | 50 | 0.0053 |
| 1346000 | 89.8 | 92.4 | 80 | 95/95 | 107.916 | 20 | 0.0054 |
| 1347000 | 92.4 | 92.34 | 84 | 95/95 | 118.772 | 30 | 0.0054 |
| 1348000 | 92.0 | 91.88 | 88 | 95/95 | 119.638 | 30 | 0.0055 |
