# c51pilot-lr5e5seed1

![c51pilot-lr5e5seed1 progress](c51pilot-lr5e5seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 600000, avg score 91.8, perfect games 30%.

## Config

| setting | value |
|---|---|
| policy_name | c51pilot-lr5e5seed1 |
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

601 evals so far. Full series in [`c51pilot-lr5e5seed1_evals.json`](c51pilot-lr5e5seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 0.8 | 0.8 | 0 | 3/95 | 0.3 | 0 | 0.4 |
| 2000 | 0.9 | 0.85 | 0 | 2/95 | 0.4 | 0 | 0.4 |
| ... | | | | | | | |
| 589000 | 89.1 | 76.9 | 65 | 95/95 | 137.45 | 50 | 0.0034 |
| 590000 | 94.6 | 85.54 | 93 | 95/95 | 173.7 | 80 | 0.0034 |
| 591000 | 89.5 | 86.32 | 68 | 95/95 | 148.7 | 60 | 0.0035 |
| 592000 | 87.1 | 87.16 | 16 | 95/95 | 176.15 | 90 | 0.0033 |
| 593000 | 85.2 | 89.1 | 47 | 95/95 | 153.0 | 70 | 0.0033 |
| 594000 | 92.8 | 89.84 | 73 | 95/95 | 181.4 | 90 | 0.0033 |
| 595000 | 84.6 | 87.84 | 19 | 95/95 | 132.95 | 50 | 0.0034 |
| 596000 | 90.7 | 88.08 | 73 | 95/95 | 138.6 | 50 | 0.0035 |
| 597000 | 75.2 | 85.7 | 39 | 95/95 | 102.3 | 30 | 0.0036 |
| 598000 | 69.6 | 82.58 | 17 | 95/95 | 86.3 | 20 | 0.0038 |
| 599000 | 85.8 | 81.18 | 61 | 95/95 | 123.75 | 40 | 0.0038 |
| 600000 | 91.8 | 82.62 | 87 | 95/95 | 121.15 | 30 | 0.0038 |
