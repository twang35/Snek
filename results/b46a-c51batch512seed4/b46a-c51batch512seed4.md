# b46a-c51batch512seed4

![b46a-c51batch512seed4 progress](b46a-c51batch512seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 91.65, perfect games 70%.

## Config

| setting | value |
|---|---|
| policy_name | b46a-c51batch512seed4 |
| seed | 4 |
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

3001 evals so far. Full series in [`b46a-c51batch512seed4_evals.json`](b46a-c51batch512seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.05 | 0.05 | 0 | 1/95 | -4.95 | 0 | 0.4 |
| 1000 | 1.65 | 1.65 | 0 | 7/95 | 1.15 | 0 | 0.4 |
| 2000 | 2.45 | 2.05 | 1 | 9/95 | 1.95 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 86.1 | 89.08 | 32 | 95/95 | 154.575 | 70 | 0.0027 |
| 2990000 | 91.65 | 89.73 | 37 | 95/95 | 170.3 | 80 | 0.0027 |
| 2991000 | 83.95 | 87.74 | 47 | 95/95 | 121.45 | 40 | 0.0028 |
| 2992000 | 92.8 | 88.61 | 64 | 95/95 | 161.5 | 70 | 0.0027 |
| 2993000 | 93.15 | 89.53 | 70 | 95/95 | 157.1 | 65 | 0.0028 |
| 2994000 | 92.45 | 90.8 | 55 | 95/95 | 161.6 | 70 | 0.0028 |
| 2995000 | 94.05 | 91.28 | 91 | 95/95 | 157.775 | 65 | 0.0028 |
| 2996000 | 92.2 | 92.93 | 50 | 95/95 | 160.9 | 70 | 0.0028 |
| 2997000 | 93.8 | 93.13 | 85 | 95/95 | 153.0 | 60 | 0.0028 |
| 2998000 | 92.65 | 93.03 | 64 | 95/95 | 156.375 | 65 | 0.0028 |
| 2999000 | 90.85 | 92.71 | 49 | 95/95 | 154.575 | 65 | 0.0028 |
| 3000000 | 91.65 | 92.23 | 67 | 95/95 | 160.575 | 70 | 0.0028 |
