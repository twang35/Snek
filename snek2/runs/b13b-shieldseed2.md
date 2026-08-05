# b13b-shieldseed2

![b13b-shieldseed2 progress](b13b-shieldseed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3700000, avg score 93.0, perfect games 50%.

## Config

| setting | value |
|---|---|
| policy_name | b13b-shieldseed2 |
| seed | 2 |
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
| epsilon_schedule | bootstrap on avg_reward [2, 5, 10, 15, 20] then geometric to floor by 80% trailing-30 perfect |
| guided_fraction | 0.5 |
| exploration_shield | 50% of refinement-phase episodes draw the epsilon move from non-fatal actions; greedy moves never shielded |
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

3701 evals so far. Full series in [`b13b-shieldseed2_evals.json`](b13b-shieldseed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.003 | 0 | 0.4 |
| 1000 | 0.8 | 0.8 | 0 | 3/95 | -0.201 | 0 | 0.4 |
| 2000 | 0.7 | 0.75 | 0 | 2/95 | 0.146 | 0 | 0.4 |
| ... | | | | | | | |
| 3689000 | 91.4 | 93.02 | 64 | 95/95 | 169.795 | 80 | 0.0028 |
| 3690000 | 92.9 | 93.0 | 90 | 95/95 | 141.268 | 50 | 0.0028 |
| 3691000 | 92.9 | 92.82 | 88 | 95/95 | 161.343 | 70 | 0.0028 |
| 3692000 | 90.8 | 92.24 | 64 | 95/95 | 149.309 | 60 | 0.0028 |
| 3693000 | 89.8 | 91.56 | 66 | 95/95 | 138.324 | 50 | 0.0029 |
| 3694000 | 91.0 | 91.48 | 64 | 95/95 | 149.48 | 60 | 0.0029 |
| 3695000 | 89.1 | 90.72 | 64 | 95/95 | 136.35 | 50 | 0.0029 |
| 3696000 | 95.0 | 91.14 | 95 | 95/95 | 193.387 | 100 | 0.0028 |
| 3697000 | 94.1 | 91.8 | 90 | 95/95 | 162.225 | 70 | 0.0028 |
| 3698000 | 94.2 | 92.68 | 90 | 95/95 | 172.623 | 80 | 0.0027 |
| 3699000 | 94.1 | 93.3 | 92 | 95/95 | 151.212 | 60 | 0.0027 |
| 3700000 | 93.0 | 94.08 | 88 | 95/95 | 141.052 | 50 | 0.0027 |
