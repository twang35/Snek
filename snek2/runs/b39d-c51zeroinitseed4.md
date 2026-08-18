# b39d-c51zeroinitseed4

![b39d-c51zeroinitseed4 progress](b39d-c51zeroinitseed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 37000, avg score 72.9, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b39d-c51zeroinitseed4 |
| seed | 4 |
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

38 evals so far. Full series in [`b39d-c51zeroinitseed4_evals.json`](b39d-c51zeroinitseed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.3 | 0.3 | 0 | 1/95 | -4.7 | 0 | 0.4 |
| 1000 | 1.2 | 1.2 | 0 | 3/95 | 0.7 | 0 | 0.4 |
| 2000 | 2.8 | 2.0 | 0 | 8/95 | 2.3 | 0 | 0.4 |
| ... | | | | | | | |
| 26000 | 77.8 | 76.68 | 32 | 95/95 | 84.55 | 10 | 0.0106 |
| 27000 | 78.4 | 78.86 | 63 | 90/95 | 77.0 | 0 | 0.0107 |
| 28000 | 74.6 | 78.28 | 58 | 89/95 | 72.3 | 0 | 0.0108 |
| 29000 | 81.2 | 78.72 | 60 | 93/95 | 79.8 | 0 | 0.0108 |
| 30000 | 85.0 | 79.4 | 77 | 91/95 | 84.5 | 0 | 0.0108 |
| 31000 | 78.2 | 79.48 | 55 | 89/95 | 77.25 | 0 | 0.0108 |
| 32000 | 58.2 | 75.44 | 0 | 88/95 | 57.25 | 0 | 0.0108 |
| 33000 | 76.1 | 75.74 | 57 | 87/95 | 75.6 | 0 | 0.0108 |
| 34000 | 85.4 | 76.58 | 79 | 90/95 | 84.45 | 0 | 0.011 |
| 35000 | 82.7 | 76.12 | 74 | 88/95 | 81.3 | 0 | 0.011 |
| 36000 | 80.5 | 76.58 | 52 | 88/95 | 80.0 | 0 | 0.011 |
| 37000 | 72.9 | 79.52 | 5 | 94/95 | 71.5 | 0 | 0.011 |
