# c51pilot-lr5e5seed2

![c51pilot-lr5e5seed2 progress](c51pilot-lr5e5seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 600000, avg score 94.3, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | c51pilot-lr5e5seed2 |
| seed | 2 |
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
| max_steps | 600000 |
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

601 evals so far. Full series in [`c51pilot-lr5e5seed2_evals.json`](c51pilot-lr5e5seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 1.1 | 1.1 | 0 | 2/95 | 0.6 | 0 | 0.4 |
| 2000 | 0.9 | 1.0 | 0 | 2/95 | 0.4 | 0 | 0.4 |
| ... | | | | | | | |
| 589000 | 92.3 | 90.94 | 77 | 95/95 | 139.75 | 50 | 0.0041 |
| 590000 | 94.4 | 91.04 | 92 | 95/95 | 162.2 | 70 | 0.004 |
| 591000 | 86.2 | 91.68 | 22 | 95/95 | 114.2 | 30 | 0.0041 |
| 592000 | 93.7 | 91.92 | 87 | 95/95 | 161.95 | 70 | 0.0039 |
| 593000 | 88.0 | 90.92 | 69 | 95/95 | 96.1 | 10 | 0.0041 |
| 594000 | 94.4 | 91.34 | 93 | 95/95 | 163.55 | 70 | 0.004 |
| 595000 | 85.4 | 89.54 | 31 | 95/95 | 123.35 | 40 | 0.0041 |
| 596000 | 85.2 | 89.34 | 7 | 95/95 | 133.55 | 50 | 0.0041 |
| 597000 | 72.9 | 85.18 | 26 | 95/95 | 120.35 | 50 | 0.0041 |
| 598000 | 93.5 | 86.28 | 90 | 95/95 | 142.75 | 50 | 0.0041 |
| 599000 | 89.3 | 85.26 | 44 | 95/95 | 137.2 | 50 | 0.0041 |
| 600000 | 94.3 | 87.04 | 93 | 95/95 | 152.15 | 60 | 0.004 |
