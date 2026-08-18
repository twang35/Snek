# b39c-c51zeroinitseed3

![b39c-c51zeroinitseed3 progress](b39c-c51zeroinitseed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 211000, avg score 94.8, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b39c-c51zeroinitseed3 |
| seed | 3 |
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

212 evals so far. Full series in [`b39c-c51zeroinitseed3_evals.json`](b39c-c51zeroinitseed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 1.6 | 1.6 | 0 | 4/95 | 1.1 | 0 | 0.4 |
| 2000 | 0.7 | 1.15 | 0 | 4/95 | 0.2 | 0 | 0.4 |
| ... | | | | | | | |
| 200000 | 93.9 | 93.58 | 90 | 95/95 | 153.1 | 60 | 0.0034 |
| 201000 | 92.5 | 93.16 | 72 | 95/95 | 171.6 | 80 | 0.0033 |
| 202000 | 95.0 | 93.64 | 95 | 95/95 | 194.0 | 100 | 0.0032 |
| 203000 | 93.1 | 93.34 | 82 | 95/95 | 151.85 | 60 | 0.0031 |
| 204000 | 90.9 | 93.08 | 58 | 95/95 | 160.05 | 70 | 0.003 |
| 205000 | 92.6 | 92.82 | 78 | 95/95 | 161.3 | 70 | 0.003 |
| 206000 | 92.9 | 92.9 | 90 | 95/95 | 131.75 | 40 | 0.003 |
| 207000 | 93.4 | 92.58 | 86 | 95/95 | 152.6 | 60 | 0.003 |
| 208000 | 81.4 | 90.24 | 26 | 95/95 | 130.65 | 50 | 0.003 |
| 209000 | 95.0 | 91.06 | 95 | 95/95 | 194.0 | 100 | 0.0028 |
| 210000 | 93.0 | 91.14 | 77 | 95/95 | 172.1 | 80 | 0.0028 |
| 211000 | 94.8 | 91.52 | 93 | 95/95 | 183.85 | 90 | 0.0027 |
