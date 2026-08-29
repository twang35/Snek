# b31a-c51lr5e5seed1

![b31a-c51lr5e5seed1 progress](b31a-c51lr5e5seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 555000, avg score 65.9, perfect games 50%.

## Config

| setting | value |
|---|---|
| policy_name | b31a-c51lr5e5seed1 |
| seed | 1 |
| zeroed_observations | none |
| learning_rate | 5e-05 |
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
| fc_layer_params | (200, 100, 100) |
| algo | c51 (distributional), 51 atoms over [-5.0, 120.0] at 2.500 spacing, cross-entropy loss, double (online argmax) target selection, standard init |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | kl (SNEK_PRIORITY_SIGNAL=td_error; a distributional agent has no TD error) |
| importance_sampling_beta | disabled |
| max_steps | 2000000 |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.0 |
| CHASE_SAFE_SHAPING | off |
| eval_only | False |
| min_checkpoint_score | 40.0 |
| c51_support_note | support [-5.0, 120.0] is below the derived maximum return 194.0, so a return above 120.0 would be clipped. Measured max is 105.0 (14% headroom); spacing 2.500. This is a judgement, not an error. |

## Evals

556 evals so far. Full series in [`b31a-c51lr5e5seed1_evals.json`](b31a-c51lr5e5seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 0.8 | 0.8 | 0 | 3/95 | 0.3 | 0 | 0.4 |
| 2000 | 0.5 | 0.65 | 0 | 2/95 | 0.0 | 0 | 0.4 |
| ... | | | | | | | |
| 544000 | 57.7 | 64.92 | 0 | 95/95 | 86.15 | 30 | 0.0037 |
| 545000 | 30.4 | 60.92 | 0 | 92/95 | 28.55 | 0 | 0.0039 |
| 546000 | 53.8 | 58.38 | 0 | 95/95 | 81.8 | 30 | 0.0041 |
| 547000 | 45.5 | 53.26 | 0 | 95/95 | 83.45 | 40 | 0.0041 |
| 548000 | 63.2 | 50.12 | 0 | 95/95 | 112.0 | 50 | 0.0042 |
| 549000 | 74.8 | 53.54 | 0 | 95/95 | 133.55 | 60 | 0.0042 |
| 550000 | 53.0 | 58.06 | 0 | 95/95 | 91.4 | 40 | 0.0043 |
| 551000 | 65.5 | 60.4 | 0 | 95/95 | 83.1 | 20 | 0.0045 |
| 552000 | 66.8 | 64.66 | 2 | 95/95 | 95.25 | 30 | 0.0045 |
| 553000 | 64.4 | 64.9 | 1 | 95/95 | 101.9 | 40 | 0.0045 |
| 554000 | 58.2 | 61.58 | 0 | 95/95 | 77.15 | 20 | 0.0046 |
| 555000 | 65.9 | 64.16 | 2 | 95/95 | 114.7 | 50 | 0.0047 |
