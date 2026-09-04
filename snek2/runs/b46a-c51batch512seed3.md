# b46a-c51batch512seed3

![b46a-c51batch512seed3 progress](b46a-c51batch512seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 93.3, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | b46a-c51batch512seed3 |
| seed | 3 |
| zeroed_observations | none |
| learning_rate | 0.0001 |
| adam_epsilon | 0.0003125 |
| perfect_game_reward | 100.0 |
| batch_size | 512 |
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
| algo | c51 (distributional), 51 atoms over [-5.0, 120.0] at 2.500 spacing, cross-entropy loss, double (online argmax) target selection, standard init |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | kl (SNEK_PRIORITY_SIGNAL=td_error; a distributional agent has no TD error) |
| importance_sampling_beta | disabled |
| max_steps | 3000000 |
| initial_populate_steps | 1000 |
| eval | 20 episodes every 1000 steps |
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

3001 evals so far. Full series in [`b46a-c51batch512seed3_evals.json`](b46a-c51batch512seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 1.2 | 1.2 | 0 | 11/95 | -3.8 | 0 | 0.4 |
| 1000 | 1.75 | 1.75 | 0 | 11/95 | 1.25 | 0 | 0.4 |
| 2000 | 1.35 | 1.55 | 0 | 3/95 | 0.85 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.1 | 93.52 | 91 | 95/95 | 157.6 | 65 | 0.0031 |
| 2990000 | 94.8 | 93.65 | 93 | 95/95 | 183.625 | 90 | 0.003 |
| 2991000 | 94.35 | 93.92 | 90 | 95/95 | 168.25 | 75 | 0.0031 |
| 2992000 | 94.45 | 94.29 | 93 | 95/95 | 163.375 | 70 | 0.0031 |
| 2993000 | 93.2 | 94.18 | 71 | 95/95 | 156.475 | 65 | 0.003 |
| 2994000 | 92.1 | 93.78 | 59 | 95/95 | 145.2 | 55 | 0.003 |
| 2995000 | 93.6 | 93.54 | 79 | 95/95 | 157.325 | 65 | 0.003 |
| 2996000 | 94.2 | 93.51 | 91 | 95/95 | 162.225 | 70 | 0.0031 |
| 2997000 | 93.85 | 93.39 | 91 | 95/95 | 147.175 | 55 | 0.0031 |
| 2998000 | 92.75 | 93.3 | 79 | 95/95 | 140.875 | 50 | 0.0031 |
| 2999000 | 92.35 | 93.35 | 78 | 95/95 | 125.325 | 35 | 0.0032 |
| 3000000 | 93.3 | 93.29 | 77 | 95/95 | 151.6 | 60 | 0.0032 |
