# b16a-noshapeseed1

![b16a-noshapeseed1 progress](b16a-noshapeseed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1245000, avg score 94.5, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b16a-noshapeseed1 |
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
| max_steps | 10000000 |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.0 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

1246 evals so far. Full series in [`b16a-noshapeseed1_evals.json`](b16a-noshapeseed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -0.5 | 0 | 0.4 |
| 2000 | 0.0 | 0.0 | 0 | 0/95 | -0.5 | 0 | 0.4 |
| ... | | | | | | | |
| 1234000 | 93.9 | 93.6 | 90 | 95/95 | 162.6 | 70 | 0.0023 |
| 1235000 | 91.1 | 93.0 | 59 | 95/95 | 170.2 | 80 | 0.0023 |
| 1236000 | 95.0 | 93.42 | 95 | 95/95 | 194.0 | 100 | 0.0023 |
| 1237000 | 94.5 | 93.84 | 90 | 95/95 | 183.55 | 90 | 0.0022 |
| 1238000 | 94.7 | 93.84 | 92 | 95/95 | 183.75 | 90 | 0.0022 |
| 1239000 | 94.2 | 93.9 | 90 | 95/95 | 172.85 | 80 | 0.0022 |
| 1240000 | 94.2 | 94.52 | 92 | 95/95 | 162.9 | 70 | 0.0022 |
| 1241000 | 93.0 | 94.12 | 78 | 95/95 | 171.65 | 80 | 0.0022 |
| 1242000 | 93.4 | 93.9 | 86 | 95/95 | 172.5 | 80 | 0.0022 |
| 1243000 | 94.5 | 93.86 | 92 | 95/95 | 173.6 | 80 | 0.0022 |
| 1244000 | 94.0 | 93.82 | 88 | 95/95 | 173.1 | 80 | 0.0022 |
| 1245000 | 94.5 | 93.88 | 90 | 95/95 | 183.55 | 90 | 0.0021 |
