# b46c-c51atoms21seed2

![b46c-c51atoms21seed2 progress](b46c-c51atoms21seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.39, perfect games 71%.

## Config

| setting | value |
|---|---|
| policy_name | b46c-c51atoms21seed2 |
| seed | 2 |
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

3001 evals so far. Full series in [`b46c-c51atoms21seed2_evals.json`](b46c-c51atoms21seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 0.22 | 0.22 | 0 | 3/95 | -0.28 | 0 | 0.4 |
| 2000 | 0.8 | 0.51 | 0 | 4/95 | 0.3 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 86.95 | 90.49 | 28 | 95/95 | 103.42 | 19 | 0.0033 |
| 2990000 | 90.14 | 90.12 | 23 | 95/95 | 129.49 | 41 | 0.0034 |
| 2991000 | 93.5 | 90.43 | 10 | 95/95 | 170.295 | 78 | 0.0033 |
| 2992000 | 92.37 | 90.64 | 35 | 95/95 | 158.945 | 68 | 0.0033 |
| 2993000 | 92.86 | 91.16 | 60 | 95/95 | 157.58 | 66 | 0.0033 |
| 2994000 | 90.19 | 91.81 | 71 | 95/95 | 115.205 | 27 | 0.0034 |
| 2995000 | 92.27 | 92.24 | 5 | 95/95 | 172.955 | 82 | 0.0034 |
| 2996000 | 92.6 | 92.06 | 45 | 95/95 | 147.235 | 56 | 0.0034 |
| 2997000 | 92.48 | 92.08 | 17 | 95/95 | 158.06 | 67 | 0.0033 |
| 2998000 | 91.27 | 91.76 | 53 | 95/95 | 147.625 | 58 | 0.0033 |
| 2999000 | 93.21 | 92.37 | 5 | 95/95 | 175.16 | 83 | 0.0033 |
| 3000000 | 93.39 | 92.59 | 63 | 95/95 | 162.95 | 71 | 0.0033 |
