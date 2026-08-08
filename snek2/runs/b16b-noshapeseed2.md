# b16b-noshapeseed2

![b16b-noshapeseed2 progress](b16b-noshapeseed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 902000, avg score 94.7, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b16b-noshapeseed2 |
| seed | 2 |
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

903 evals so far. Full series in [`b16b-noshapeseed2_evals.json`](b16b-noshapeseed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.8 | 0.8 | 0 | 3/95 | -0.15 | 0 | 0.4 |
| 2000 | 0.7 | 0.75 | 0 | 2/95 | 0.2 | 0 | 0.4 |
| ... | | | | | | | |
| 891000 | 95.0 | 94.72 | 95 | 95/95 | 194.0 | 100 | 0.0021 |
| 892000 | 94.5 | 94.7 | 92 | 95/95 | 162.75 | 70 | 0.0021 |
| 893000 | 92.9 | 94.32 | 77 | 95/95 | 172.0 | 80 | 0.0022 |
| 894000 | 93.9 | 94.26 | 87 | 95/95 | 173.0 | 80 | 0.0022 |
| 895000 | 95.0 | 94.26 | 95 | 95/95 | 194.0 | 100 | 0.0021 |
| 896000 | 94.6 | 94.18 | 92 | 95/95 | 173.25 | 80 | 0.0021 |
| 897000 | 94.7 | 94.22 | 92 | 95/95 | 183.75 | 90 | 0.0021 |
| 898000 | 94.5 | 94.54 | 92 | 95/95 | 173.15 | 80 | 0.0021 |
| 899000 | 95.0 | 94.76 | 95 | 95/95 | 194.0 | 100 | 0.0021 |
| 900000 | 92.1 | 94.18 | 66 | 95/95 | 181.15 | 90 | 0.0021 |
| 901000 | 93.8 | 94.02 | 84 | 95/95 | 172.45 | 80 | 0.002 |
| 902000 | 94.7 | 94.02 | 92 | 95/95 | 183.75 | 90 | 0.002 |
