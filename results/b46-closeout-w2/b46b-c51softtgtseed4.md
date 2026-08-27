# b46b-c51softtgtseed4

![b46b-c51softtgtseed4 progress](b46b-c51softtgtseed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.02, perfect games 52%.

Training was resumed at step 845000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | b46b-c51softtgtseed4 |
| seed | 4 |
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

3001 evals so far. Full series in [`b46b-c51softtgtseed4_evals.json`](b46b-c51softtgtseed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.05 | 0.05 | 0 | 1/95 | -4.95 | 0 | 0.4 |
| 1000 | 0.15 | 0.15 | 0 | 1/95 | -0.35 | 0 | 0.4 |
| 2000 | 0.5 | 0.33 | 0 | 3/95 | 0.0 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 86.34 | 79.34 | 23 | 95/95 | 132.655 | 48 | 0.0045 |
| 2990000 | 86.9 | 81.52 | 21 | 95/95 | 138.235 | 53 | 0.0045 |
| 2991000 | 77.19 | 80.55 | 17 | 95/95 | 112.565 | 38 | 0.0045 |
| 2992000 | 80.9 | 81.03 | 17 | 95/95 | 133.685 | 55 | 0.0044 |
| 2993000 | 89.16 | 84.1 | 15 | 95/95 | 129.19 | 42 | 0.0044 |
| 2994000 | 88.24 | 84.48 | 14 | 95/95 | 145.59 | 59 | 0.0044 |
| 2995000 | 89.77 | 85.05 | 27 | 95/95 | 145.4 | 57 | 0.0044 |
| 2996000 | 91.81 | 87.98 | 17 | 95/95 | 128.31 | 38 | 0.0044 |
| 2997000 | 93.0 | 90.4 | 38 | 95/95 | 160.885 | 69 | 0.0043 |
| 2998000 | 93.53 | 91.27 | 78 | 95/95 | 149.16 | 57 | 0.0043 |
| 2999000 | 92.47 | 92.12 | 29 | 95/95 | 128.925 | 38 | 0.0044 |
| 3000000 | 93.02 | 92.77 | 41 | 95/95 | 143.585 | 52 | 0.0043 |
