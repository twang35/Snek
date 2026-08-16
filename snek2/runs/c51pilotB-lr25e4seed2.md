# c51pilotB-lr25e4seed2

![c51pilotB-lr25e4seed2 progress](c51pilotB-lr25e4seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 600000, avg score 39.7, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | c51pilotB-lr25e4seed2 |
| seed | 2 |
| zeroed_observations | none |
| learning_rate | 0.00025 |
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

601 evals so far. Full series in [`c51pilotB-lr25e4seed2_evals.json`](c51pilotB-lr25e4seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 1.2 | 1.2 | 0 | 3/95 | 0.7 | 0 | 0.4 |
| 2000 | 2.7 | 1.95 | 0 | 7/95 | 2.2 | 0 | 0.4 |
| ... | | | | | | | |
| 589000 | 24.5 | 31.12 | 0 | 67/95 | 19.95 | 0 | 0.0123 |
| 590000 | 31.6 | 26.68 | 1 | 65/95 | 27.5 | 0 | 0.0123 |
| 591000 | 45.3 | 34.22 | 0 | 58/95 | 40.75 | 0 | 0.0123 |
| 592000 | 33.0 | 35.7 | 1 | 85/95 | 28.45 | 0 | 0.0123 |
| 593000 | 47.3 | 36.34 | 33 | 63/95 | 42.3 | 0 | 0.0123 |
| 594000 | 39.4 | 39.32 | 0 | 76/95 | 34.4 | 0 | 0.0123 |
| 595000 | 51.9 | 43.38 | 4 | 91/95 | 46.9 | 0 | 0.0123 |
| 596000 | 62.8 | 46.88 | 12 | 95/95 | 69.1 | 10 | 0.0122 |
| 597000 | 42.7 | 48.82 | 1 | 77/95 | 37.7 | 0 | 0.0122 |
| 598000 | 61.7 | 51.7 | 39 | 77/95 | 57.6 | 0 | 0.0122 |
| 599000 | 67.8 | 57.38 | 22 | 95/95 | 74.55 | 10 | 0.0121 |
| 600000 | 39.7 | 54.94 | 0 | 92/95 | 35.15 | 0 | 0.0121 |
