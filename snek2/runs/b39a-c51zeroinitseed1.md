# b39a-c51zeroinitseed1

![b39a-c51zeroinitseed1 progress](b39a-c51zeroinitseed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 46000, avg score 82.2, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b39a-c51zeroinitseed1 |
| seed | 1 |
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

47 evals so far. Full series in [`b39a-c51zeroinitseed1_evals.json`](b39a-c51zeroinitseed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.2 | 0.2 | 0 | 1/95 | -4.8 | 0 | 0.4 |
| 1000 | 1.4 | 1.4 | 0 | 3/95 | 0.9 | 0 | 0.4 |
| 2000 | 2.3 | 1.85 | 0 | 8/95 | 1.8 | 0 | 0.4 |
| ... | | | | | | | |
| 35000 | 88.1 | 81.94 | 76 | 95/95 | 96.2 | 10 | 0.0103 |
| 36000 | 82.1 | 81.32 | 57 | 95/95 | 100.15 | 20 | 0.0102 |
| 37000 | 89.9 | 85.48 | 82 | 95/95 | 138.7 | 50 | 0.0098 |
| 38000 | 69.4 | 82.14 | 19 | 95/95 | 75.7 | 10 | 0.0097 |
| 39000 | 86.3 | 83.16 | 51 | 95/95 | 102.55 | 20 | 0.0096 |
| 40000 | 63.8 | 78.3 | 22 | 95/95 | 69.2 | 10 | 0.0095 |
| 41000 | 77.8 | 77.44 | 19 | 95/95 | 85.0 | 10 | 0.0094 |
| 42000 | 92.1 | 77.88 | 80 | 95/95 | 140.0 | 50 | 0.0091 |
| 43000 | 82.1 | 80.42 | 39 | 95/95 | 89.75 | 10 | 0.009 |
| 44000 | 83.8 | 79.92 | 37 | 95/95 | 91.0 | 10 | 0.0089 |
| 45000 | 71.2 | 81.4 | 11 | 94/95 | 68.45 | 0 | 0.0089 |
| 46000 | 82.2 | 82.28 | 72 | 94/95 | 80.35 | 0 | 0.009 |
