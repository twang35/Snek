# b46b-c51softtgtseed1

![b46b-c51softtgtseed1 progress](b46b-c51softtgtseed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.32, perfect games 73%.

Training was resumed at step 821000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | b46b-c51softtgtseed1 |
| seed | 1 |
| zeroed_observations | none |
| learning_rate | 0.0001 |
| adam_epsilon | 0.0003125 |
| perfect_game_reward | 100.0 |
| batch_size | 128 |
| discount | 0.9975 |
| target_update_period | 1 |
| target_update_tau | 0.005 |
| gradient_clipping | none |
| n_step_update | 1 |
| initial_epsilon | 0.4 |
| min_epsilon | 0.002 |
| epsilon_schedule | bootstrap on avg_reward [2, 5, 10, 15, 20] then geometric to floor by 80% trailing-30 perfect |
| guided_fraction | 0.8 |
| forking | up to 4 live branches including the main line, fork p=0.5 at length >= 85, branch capped at 60 steps, one branch advanced per iteration |
| exploration_shield | 80% of refinement-phase episodes draw the epsilon move from non-fatal actions; greedy moves never shielded |
| fc_layer_params | (320,) |
| algo | c51 (distributional), 51 atoms over [-5.0, 120.0] at 2.500 spacing, cross-entropy loss, double (online argmax) target selection, standard init |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | kl (SNEK_PRIORITY_SIGNAL=td_error; a distributional agent has no TD error) |
| importance_sampling_beta | disabled |
| max_steps | 3000000 |
| initial_populate_steps | 1000 |
| eval | 100 episodes every 1000 steps, engine vec, 100 lanes in-process, no worker processes |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.0 |
| CHASE_SAFE_SHAPING | off |
| FREE_SPACE_SHAPING | off |
| eval_only | False |
| min_checkpoint_score | 40.0 |
| c51_support_note | support [-5.0, 120.0] is below the derived maximum return 194.0, so a return above 120.0 would be clipped. 14% headroom over the measured 105.0; spacing 2.500. This is a judgement, not an error. |

## Evals

3001 evals so far. Full series in [`b46b-c51softtgtseed1_evals.json`](b46b-c51softtgtseed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.9 | 0.9 | 0 | 3/95 | 0.4 | 0 | 0.4 |
| 2000 | 1.3 | 1.1 | 0 | 8/95 | 0.8 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 90.95 | 91.03 | 10 | 95/95 | 151.465 | 62 | 0.004 |
| 2990000 | 91.79 | 91.14 | 14 | 95/95 | 159.585 | 69 | 0.004 |
| 2991000 | 93.49 | 91.86 | 39 | 95/95 | 156.22 | 64 | 0.0039 |
| 2992000 | 90.96 | 91.45 | 20 | 95/95 | 156.36 | 67 | 0.0039 |
| 2993000 | 92.14 | 91.87 | 35 | 95/95 | 136.285 | 46 | 0.0039 |
| 2994000 | 91.87 | 92.05 | 31 | 95/95 | 155.505 | 65 | 0.0037 |
| 2995000 | 92.52 | 92.2 | 21 | 95/95 | 142.635 | 52 | 0.0037 |
| 2996000 | 93.38 | 92.17 | 49 | 95/95 | 161.9 | 70 | 0.0037 |
| 2997000 | 91.13 | 92.21 | 19 | 95/95 | 158.61 | 69 | 0.0035 |
| 2998000 | 93.57 | 92.49 | 44 | 95/95 | 153.045 | 61 | 0.0035 |
| 2999000 | 91.15 | 92.35 | 12 | 95/95 | 139.91 | 51 | 0.0035 |
| 3000000 | 93.32 | 92.51 | 45 | 95/95 | 165.05 | 73 | 0.0035 |
