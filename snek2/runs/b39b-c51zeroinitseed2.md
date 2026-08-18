# b39b-c51zeroinitseed2

![b39b-c51zeroinitseed2 progress](b39b-c51zeroinitseed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 42000, avg score 66.5, perfect games 20%.

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

43 evals so far. Full series in [`b39b-c51zeroinitseed2_evals.json`](b39b-c51zeroinitseed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.3 | 0.3 | 0 | 2/95 | -0.2 | 0 | 0.4 |
| 2000 | 0.7 | 0.5 | 0 | 2/95 | 0.2 | 0 | 0.4 |
| ... | | | | | | | |
| 31000 | 68.8 | 70.32 | 51 | 82/95 | 65.15 | 0 | 0.0119 |
| 32000 | 75.9 | 71.42 | 43 | 91/95 | 71.8 | 0 | 0.0119 |
| 33000 | 77.2 | 71.08 | 61 | 83/95 | 74.9 | 0 | 0.0119 |
| 34000 | 75.3 | 72.24 | 51 | 89/95 | 71.2 | 0 | 0.0119 |
| 35000 | 80.4 | 75.52 | 58 | 89/95 | 76.3 | 0 | 0.0119 |
| 36000 | 75.9 | 76.94 | 56 | 92/95 | 73.15 | 0 | 0.0119 |
| 37000 | 74.9 | 76.74 | 54 | 84/95 | 70.8 | 0 | 0.0119 |
| 38000 | 69.6 | 75.22 | 27 | 88/95 | 65.95 | 0 | 0.0119 |
| 39000 | 67.8 | 73.72 | 34 | 92/95 | 63.25 | 0 | 0.0119 |
| 40000 | 67.3 | 71.1 | 25 | 93/95 | 62.75 | 0 | 0.0119 |
| 41000 | 59.0 | 67.72 | 24 | 90/95 | 54.45 | 0 | 0.0119 |
| 42000 | 66.5 | 66.04 | 17 | 95/95 | 83.65 | 20 | 0.0118 |
