# c51pilotB-lr25e4seed1

![c51pilotB-lr25e4seed1 progress](c51pilotB-lr25e4seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 600000, avg score 1.1, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | c51pilotB-lr25e4seed1 |
| seed | 1 |
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

601 evals so far. Full series in [`c51pilotB-lr25e4seed1_evals.json`](c51pilotB-lr25e4seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 0.8 | 0.8 | 0 | 3/95 | 0.3 | 0 | 0.4 |
| 2000 | 2.1 | 1.45 | 0 | 4/95 | 1.6 | 0 | 0.4 |
| ... | | | | | | | |
| 589000 | 0.9 | 0.52 | 0 | 4/95 | -0.5 | 0 | 0.4 |
| 590000 | 0.5 | 0.48 | 0 | 1/95 | -1.8 | 0 | 0.4 |
| 591000 | 0.3 | 0.5 | 0 | 2/95 | -2.45 | 0 | 0.4 |
| 592000 | 0.9 | 0.6 | 0 | 3/95 | -0.5 | 0 | 0.4 |
| 593000 | 0.7 | 0.66 | 0 | 2/95 | -0.25 | 0 | 0.4 |
| 594000 | 0.5 | 0.58 | 0 | 3/95 | -2.7 | 0 | 0.4 |
| 595000 | 1.2 | 0.72 | 0 | 3/95 | -0.2 | 0 | 0.4 |
| 596000 | 1.5 | 0.96 | 0 | 6/95 | 0.1 | 0 | 0.4 |
| 597000 | 1.4 | 1.06 | 0 | 5/95 | -0.9 | 0 | 0.4 |
| 598000 | 0.5 | 1.02 | 0 | 2/95 | -1.8 | 0 | 0.4 |
| 599000 | 0.1 | 0.94 | 0 | 1/95 | -2.65 | 0 | 0.4 |
| 600000 | 1.1 | 0.92 | 0 | 3/95 | -0.75 | 0 | 0.4 |
