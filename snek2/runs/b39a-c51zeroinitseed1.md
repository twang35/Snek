# b39a-c51zeroinitseed1

![b39a-c51zeroinitseed1 progress](b39a-c51zeroinitseed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 92.8, perfect games 60%.

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

3001 evals so far. Full series in [`b39a-c51zeroinitseed1_evals.json`](b39a-c51zeroinitseed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.2 | 0.2 | 0 | 1/95 | -4.8 | 0 | 0.4 |
| 1000 | 1.4 | 1.4 | 0 | 3/95 | 0.9 | 0 | 0.4 |
| 2000 | 2.3 | 1.85 | 0 | 8/95 | 1.8 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 90.3 | 91.76 | 65 | 95/95 | 119.2 | 30 | 0.0048 |
| 2990000 | 90.7 | 91.28 | 74 | 95/95 | 110.1 | 20 | 0.0047 |
| 2991000 | 91.6 | 91.12 | 69 | 95/95 | 150.35 | 60 | 0.0047 |
| 2992000 | 87.4 | 90.52 | 71 | 95/95 | 134.85 | 50 | 0.0047 |
| 2993000 | 95.0 | 91.0 | 95 | 95/95 | 194.0 | 100 | 0.0045 |
| 2994000 | 87.1 | 90.36 | 26 | 95/95 | 135.45 | 50 | 0.0045 |
| 2995000 | 93.5 | 90.92 | 91 | 95/95 | 141.85 | 50 | 0.0045 |
| 2996000 | 85.3 | 89.66 | 69 | 95/95 | 102.9 | 20 | 0.0046 |
| 2997000 | 94.0 | 90.98 | 91 | 95/95 | 151.85 | 60 | 0.0046 |
| 2998000 | 92.9 | 90.56 | 81 | 95/95 | 151.2 | 60 | 0.0045 |
| 2999000 | 91.9 | 91.52 | 75 | 95/95 | 140.7 | 50 | 0.0045 |
| 3000000 | 92.8 | 91.38 | 81 | 95/95 | 150.2 | 60 | 0.0045 |
