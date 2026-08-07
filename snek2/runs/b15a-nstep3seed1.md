# b15a-nstep3seed1

![b15a-nstep3seed1 progress](b15a-nstep3seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 5788000, avg score 88.2, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | b15a-nstep3seed1 |
| seed | 1 |
| zeroed_observations | none |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.995 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 3 |
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
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

5789 evals so far. Full series in [`b15a-nstep3seed1_evals.json`](b15a-nstep3seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.002 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -0.551 | 0 | 0.4 |
| 2000 | 0.0 | 0.0 | 0 | 0/95 | -5.003 | 0 | 0.4 |
| ... | | | | | | | |
| 5777000 | 93.8 | 92.7 | 86 | 95/95 | 172.33 | 80 | 0.0025 |
| 5778000 | 94.4 | 94.36 | 92 | 95/95 | 172.929 | 80 | 0.0025 |
| 5779000 | 94.0 | 94.34 | 88 | 95/95 | 172.486 | 80 | 0.0026 |
| 5780000 | 90.3 | 93.5 | 53 | 95/95 | 167.997 | 80 | 0.0026 |
| 5781000 | 83.2 | 91.14 | 12 | 95/95 | 151.415 | 70 | 0.0026 |
| 5782000 | 85.2 | 89.42 | 43 | 95/95 | 152.996 | 70 | 0.0027 |
| 5783000 | 93.2 | 89.18 | 80 | 95/95 | 171.319 | 80 | 0.0027 |
| 5784000 | 85.6 | 87.5 | 24 | 95/95 | 153.778 | 70 | 0.0027 |
| 5785000 | 91.1 | 87.66 | 69 | 95/95 | 159.265 | 70 | 0.0028 |
| 5786000 | 94.6 | 89.94 | 92 | 95/95 | 172.654 | 80 | 0.0027 |
| 5787000 | 90.6 | 91.02 | 54 | 95/95 | 158.75 | 70 | 0.0027 |
| 5788000 | 88.2 | 90.02 | 45 | 95/95 | 146.379 | 60 | 0.0027 |
