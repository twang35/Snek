# b46c-c51atoms21seed3

![b46c-c51atoms21seed3 progress](b46c-c51atoms21seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 91.69, perfect games 52%.

## Config

| setting | value |
|---|---|
| policy_name | b46c-c51atoms21seed3 |
| seed | 3 |
| zeroed_observations | none |
| learning_rate | 0.0001 |
| adam_epsilon | 0.0003125 |
| perfect_game_reward | 100.0 |
| batch_size | 128 |
| discount | 0.9975 |
| target_update_period | 1000 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 1 |
| initial_epsilon | 0.4 |
| min_epsilon | 0.002 |
| epsilon_schedule | bootstrap on avg_reward [2, 5, 10, 15, 20] then geometric to floor by 80% trailing-30 perfect |
| guided_fraction | 0.8 |
| forking | up to 4 live branches including the main line, fork p=0.5 at length >= 85, branch capped at 60 steps, one branch advanced per iteration |
| exploration_shield | 80% of refinement-phase episodes draw the epsilon move from non-fatal actions; greedy moves never shielded |
| fc_layer_params | (320,) |
| algo | c51 (distributional), 21 atoms over [-5.0, 120.0] at 6.250 spacing, cross-entropy loss, double (online argmax) target selection, standard init |
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
| c51_support_note | support [-5.0, 120.0] is below the derived maximum return 194.0, so a return above 120.0 would be clipped. 14% headroom over the measured 105.0; spacing 6.250. This is a judgement, not an error. |

## Evals

3001 evals so far. Full series in [`b46c-c51atoms21seed3_evals.json`](b46c-c51atoms21seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.18 | 0.18 | 0 | 2/95 | -4.82 | 0 | 0.4 |
| 1000 | 0.65 | 0.65 | 0 | 4/95 | 0.15 | 0 | 0.4 |
| 2000 | 0.6 | 0.62 | 0 | 4/95 | 0.1 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 90.09 | 90.71 | 43 | 95/95 | 135.82 | 48 | 0.0042 |
| 2990000 | 89.06 | 90.18 | 59 | 95/95 | 111.865 | 26 | 0.0043 |
| 2991000 | 91.5 | 90.1 | 65 | 95/95 | 133.385 | 44 | 0.0043 |
| 2992000 | 91.24 | 90.49 | 55 | 95/95 | 137.605 | 49 | 0.0044 |
| 2993000 | 92.13 | 90.8 | 59 | 95/95 | 139.08 | 49 | 0.0044 |
| 2994000 | 92.8 | 91.35 | 79 | 95/95 | 139.66 | 49 | 0.0043 |
| 2995000 | 90.33 | 91.6 | 43 | 95/95 | 145.375 | 57 | 0.0044 |
| 2996000 | 91.74 | 91.65 | 61 | 95/95 | 149.545 | 60 | 0.0043 |
| 2997000 | 86.57 | 90.71 | 53 | 95/95 | 98.885 | 16 | 0.0044 |
| 2998000 | 91.67 | 90.62 | 63 | 95/95 | 144.455 | 55 | 0.0044 |
| 2999000 | 91.53 | 90.37 | 55 | 95/95 | 149.425 | 60 | 0.0044 |
| 3000000 | 91.69 | 90.64 | 57 | 95/95 | 141.535 | 52 | 0.0043 |
