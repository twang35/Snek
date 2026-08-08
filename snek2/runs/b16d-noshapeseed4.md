# b16d-noshapeseed4

![b16d-noshapeseed4 progress](b16d-noshapeseed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 906000, avg score 93.6, perfect games 70%.

## Config

| setting | value |
|---|---|
| policy_name | b16d-noshapeseed4 |
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

907 evals so far. Full series in [`b16d-noshapeseed4_evals.json`](b16d-noshapeseed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 0 | 1/95 | -4.5 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 3/95 | 0.2 | 0 | 0.4 |
| 2000 | 0.7 | 0.7 | 0 | 2/95 | 0.2 | 0 | 0.4 |
| ... | | | | | | | |
| 895000 | 94.0 | 93.86 | 92 | 95/95 | 152.75 | 60 | 0.0035 |
| 896000 | 94.2 | 93.9 | 88 | 95/95 | 172.85 | 80 | 0.0034 |
| 897000 | 94.7 | 94.14 | 93 | 95/95 | 173.35 | 80 | 0.0034 |
| 898000 | 92.9 | 93.9 | 86 | 95/95 | 142.15 | 50 | 0.0034 |
| 899000 | 94.2 | 94.0 | 91 | 95/95 | 162.9 | 70 | 0.0034 |
| 900000 | 94.2 | 94.04 | 92 | 95/95 | 163.35 | 70 | 0.0033 |
| 901000 | 93.4 | 93.88 | 88 | 95/95 | 152.6 | 60 | 0.0033 |
| 902000 | 92.2 | 93.38 | 80 | 95/95 | 109.8 | 20 | 0.0035 |
| 903000 | 93.8 | 93.56 | 92 | 95/95 | 142.6 | 50 | 0.0035 |
| 904000 | 93.9 | 93.5 | 92 | 95/95 | 131.85 | 40 | 0.0035 |
| 905000 | 94.0 | 93.46 | 92 | 95/95 | 152.75 | 60 | 0.0035 |
| 906000 | 93.6 | 93.5 | 84 | 95/95 | 162.3 | 70 | 0.0035 |
