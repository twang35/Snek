# b39b-c51zeroinitseed2

![b39b-c51zeroinitseed2 progress](b39b-c51zeroinitseed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 92.7, perfect games 40%.

## Config

| setting | value |
|---|---|
| policy_name | b39b-c51zeroinitseed2 |
| seed | 2 |
| zeroed_observations | none |
| learning_rate | 0.0001 |
| adam_epsilon | 0.00015 |
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
| algo | c51 (distributional), 51 atoms over [-5.0, 120.0] at 2.500 spacing, cross-entropy loss, double (online argmax) target selection, zero-expected-Q init |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | kl (SNEK_PRIORITY_SIGNAL=td_error; a distributional agent has no TD error) |
| importance_sampling_beta | disabled |
| max_steps | 3000000 |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
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

3001 evals so far. Full series in [`b39b-c51zeroinitseed2_evals.json`](b39b-c51zeroinitseed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.3 | 0.3 | 0 | 2/95 | -0.2 | 0 | 0.4 |
| 2000 | 0.7 | 0.5 | 0 | 2/95 | 0.2 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 89.4 | 87.3 | 46 | 95/95 | 157.65 | 70 | 0.0035 |
| 2990000 | 87.9 | 88.9 | 32 | 95/95 | 145.3 | 60 | 0.0035 |
| 2991000 | 74.0 | 86.8 | 18 | 95/95 | 131.85 | 60 | 0.0034 |
| 2992000 | 67.9 | 82.64 | 25 | 95/95 | 125.3 | 60 | 0.0034 |
| 2993000 | 74.0 | 78.64 | 26 | 95/95 | 90.7 | 20 | 0.0036 |
| 2994000 | 85.9 | 77.94 | 21 | 95/95 | 123.4 | 40 | 0.0036 |
| 2995000 | 68.1 | 73.98 | 18 | 95/95 | 95.2 | 30 | 0.0036 |
| 2996000 | 86.4 | 76.46 | 41 | 95/95 | 113.5 | 30 | 0.0036 |
| 2997000 | 88.0 | 80.48 | 39 | 95/95 | 156.25 | 70 | 0.0037 |
| 2998000 | 82.6 | 82.2 | 21 | 95/95 | 119.2 | 40 | 0.0037 |
| 2999000 | 94.4 | 83.9 | 93 | 95/95 | 162.65 | 70 | 0.0037 |
| 3000000 | 92.7 | 88.82 | 86 | 95/95 | 130.65 | 40 | 0.0037 |
