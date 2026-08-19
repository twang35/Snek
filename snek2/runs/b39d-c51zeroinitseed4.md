# b39d-c51zeroinitseed4

![b39d-c51zeroinitseed4 progress](b39d-c51zeroinitseed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 94.1, perfect games 60%.

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

3001 evals so far. Full series in [`b39d-c51zeroinitseed4_evals.json`](b39d-c51zeroinitseed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.3 | 0.3 | 0 | 1/95 | -4.7 | 0 | 0.4 |
| 1000 | 1.2 | 1.2 | 0 | 3/95 | 0.7 | 0 | 0.4 |
| 2000 | 2.8 | 2.0 | 0 | 8/95 | 2.3 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.0 | 92.18 | 91 | 95/95 | 152.75 | 60 | 0.0039 |
| 2990000 | 91.8 | 91.66 | 65 | 95/95 | 170.45 | 80 | 0.0039 |
| 2991000 | 89.7 | 91.7 | 46 | 95/95 | 168.35 | 80 | 0.0038 |
| 2992000 | 87.9 | 90.96 | 66 | 95/95 | 115.0 | 30 | 0.0039 |
| 2993000 | 94.6 | 91.6 | 91 | 95/95 | 183.2 | 90 | 0.0037 |
| 2994000 | 94.0 | 91.6 | 93 | 95/95 | 143.25 | 50 | 0.0037 |
| 2995000 | 93.2 | 91.88 | 87 | 95/95 | 141.55 | 50 | 0.0037 |
| 2996000 | 93.4 | 92.62 | 91 | 95/95 | 132.25 | 40 | 0.0038 |
| 2997000 | 91.8 | 93.4 | 65 | 95/95 | 170.45 | 80 | 0.0037 |
| 2998000 | 93.8 | 93.24 | 91 | 95/95 | 142.15 | 50 | 0.0036 |
| 2999000 | 87.6 | 91.96 | 70 | 95/95 | 115.6 | 30 | 0.0037 |
| 3000000 | 94.1 | 92.14 | 92 | 95/95 | 153.3 | 60 | 0.0036 |
