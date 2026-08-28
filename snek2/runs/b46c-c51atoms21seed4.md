# b46c-c51atoms21seed4

![b46c-c51atoms21seed4 progress](b46c-c51atoms21seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 92.41, perfect games 57%.

## Config

| setting | value |
|---|---|
| policy_name | b46c-c51atoms21seed4 |
| seed | 4 |
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

3001 evals so far. Full series in [`b46c-c51atoms21seed4_evals.json`](b46c-c51atoms21seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.29 | 0.29 | 0 | 3/95 | -4.71 | 0 | 0.4 |
| 1000 | 1.04 | 1.04 | 0 | 5/95 | 0.54 | 0 | 0.4 |
| 2000 | 1.79 | 1.42 | 0 | 6/95 | 1.29 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 90.92 | 89.82 | 25 | 95/95 | 142.21 | 53 | 0.0032 |
| 2990000 | 87.78 | 89.87 | 12 | 95/95 | 134.82 | 49 | 0.0032 |
| 2991000 | 91.35 | 89.91 | 29 | 95/95 | 129.885 | 40 | 0.0033 |
| 2992000 | 89.37 | 90.25 | 21 | 95/95 | 136.41 | 49 | 0.0033 |
| 2993000 | 93.47 | 90.58 | 29 | 95/95 | 167.145 | 75 | 0.0033 |
| 2994000 | 92.07 | 90.81 | 6 | 95/95 | 147.655 | 57 | 0.0033 |
| 2995000 | 89.78 | 91.21 | 0 | 95/95 | 134.15 | 46 | 0.0033 |
| 2996000 | 90.57 | 91.05 | 0 | 95/95 | 165.24 | 76 | 0.0032 |
| 2997000 | 90.47 | 91.27 | 14 | 95/95 | 151.665 | 63 | 0.0032 |
| 2998000 | 88.21 | 90.22 | 30 | 95/95 | 145.29 | 59 | 0.0032 |
| 2999000 | 89.35 | 89.68 | 0 | 95/95 | 144.755 | 57 | 0.0032 |
| 3000000 | 92.41 | 90.2 | 69 | 95/95 | 147.635 | 57 | 0.0032 |
