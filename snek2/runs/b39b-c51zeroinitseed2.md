# b39b-c51zeroinitseed2

![b39b-c51zeroinitseed2 progress](b39b-c51zeroinitseed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 231000, avg score 93.7, perfect games 60%.

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

232 evals so far. Full series in [`b39b-c51zeroinitseed2_evals.json`](b39b-c51zeroinitseed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.3 | 0.3 | 0 | 2/95 | -0.2 | 0 | 0.4 |
| 2000 | 0.7 | 0.5 | 0 | 2/95 | 0.2 | 0 | 0.4 |
| ... | | | | | | | |
| 220000 | 93.1 | 91.38 | 88 | 95/95 | 132.4 | 40 | 0.0035 |
| 221000 | 94.4 | 91.82 | 91 | 95/95 | 173.5 | 80 | 0.0034 |
| 222000 | 93.7 | 92.12 | 89 | 95/95 | 162.85 | 70 | 0.0034 |
| 223000 | 87.0 | 90.82 | 19 | 95/95 | 155.7 | 70 | 0.0033 |
| 224000 | 94.2 | 92.48 | 91 | 95/95 | 163.35 | 70 | 0.0032 |
| 225000 | 94.8 | 92.82 | 93 | 95/95 | 183.85 | 90 | 0.0032 |
| 226000 | 94.2 | 92.78 | 89 | 95/95 | 173.3 | 80 | 0.003 |
| 227000 | 93.3 | 92.7 | 86 | 95/95 | 152.5 | 60 | 0.003 |
| 228000 | 94.2 | 94.14 | 93 | 95/95 | 153.4 | 60 | 0.0031 |
| 229000 | 94.4 | 94.18 | 93 | 95/95 | 163.55 | 70 | 0.003 |
| 230000 | 91.1 | 93.44 | 76 | 95/95 | 140.35 | 50 | 0.0031 |
| 231000 | 93.7 | 93.34 | 90 | 95/95 | 152.9 | 60 | 0.0031 |
