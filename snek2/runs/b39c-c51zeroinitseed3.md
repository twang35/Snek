# b39c-c51zeroinitseed3

![b39c-c51zeroinitseed3 progress](b39c-c51zeroinitseed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 86.0, perfect games 60%.

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

3001 evals so far. Full series in [`b39c-c51zeroinitseed3_evals.json`](b39c-c51zeroinitseed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 1.6 | 1.6 | 0 | 4/95 | 1.1 | 0 | 0.4 |
| 2000 | 0.7 | 1.15 | 0 | 4/95 | 0.2 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 91.8 | 93.24 | 79 | 95/95 | 128.85 | 40 | 0.0035 |
| 2990000 | 92.5 | 93.1 | 86 | 95/95 | 120.95 | 30 | 0.0036 |
| 2991000 | 91.5 | 92.82 | 72 | 95/95 | 140.75 | 50 | 0.0036 |
| 2992000 | 93.3 | 92.56 | 86 | 95/95 | 142.1 | 50 | 0.0036 |
| 2993000 | 93.4 | 92.5 | 91 | 95/95 | 131.35 | 40 | 0.0036 |
| 2994000 | 93.6 | 92.86 | 91 | 95/95 | 142.4 | 50 | 0.0036 |
| 2995000 | 90.9 | 92.54 | 69 | 95/95 | 139.7 | 50 | 0.0036 |
| 2996000 | 91.2 | 92.48 | 67 | 95/95 | 138.65 | 50 | 0.0037 |
| 2997000 | 94.4 | 92.7 | 93 | 95/95 | 163.55 | 70 | 0.0036 |
| 2998000 | 88.4 | 91.7 | 67 | 93/95 | 85.65 | 0 | 0.0038 |
| 2999000 | 93.8 | 91.74 | 93 | 95/95 | 131.75 | 40 | 0.0039 |
| 3000000 | 86.0 | 90.76 | 18 | 95/95 | 144.3 | 60 | 0.0039 |
