# b39d-c51zeroinitseed4

![b39d-c51zeroinitseed4 progress](b39d-c51zeroinitseed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 205000, avg score 92.6, perfect games 40%.

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

206 evals so far. Full series in [`b39d-c51zeroinitseed4_evals.json`](b39d-c51zeroinitseed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.3 | 0.3 | 0 | 1/95 | -4.7 | 0 | 0.4 |
| 1000 | 1.2 | 1.2 | 0 | 3/95 | 0.7 | 0 | 0.4 |
| 2000 | 2.8 | 2.0 | 0 | 8/95 | 2.3 | 0 | 0.4 |
| ... | | | | | | | |
| 194000 | 79.6 | 85.42 | 2 | 95/95 | 108.5 | 30 | 0.005 |
| 195000 | 85.3 | 83.62 | 20 | 95/95 | 93.4 | 10 | 0.0051 |
| 196000 | 85.1 | 85.48 | 2 | 95/95 | 143.85 | 60 | 0.005 |
| 197000 | 84.0 | 85.22 | 6 | 95/95 | 112.9 | 30 | 0.0049 |
| 198000 | 83.5 | 83.5 | 4 | 95/95 | 122.35 | 40 | 0.0049 |
| 199000 | 90.4 | 85.66 | 69 | 95/95 | 159.55 | 70 | 0.0048 |
| 200000 | 91.9 | 86.98 | 86 | 95/95 | 110.85 | 20 | 0.0049 |
| 201000 | 94.0 | 88.76 | 89 | 95/95 | 173.1 | 80 | 0.0048 |
| 202000 | 94.8 | 90.92 | 93 | 95/95 | 183.85 | 90 | 0.0046 |
| 203000 | 93.6 | 92.94 | 89 | 95/95 | 142.85 | 50 | 0.0046 |
| 204000 | 88.5 | 92.56 | 67 | 95/95 | 127.8 | 40 | 0.0046 |
| 205000 | 92.6 | 92.7 | 88 | 95/95 | 131.9 | 40 | 0.0045 |
