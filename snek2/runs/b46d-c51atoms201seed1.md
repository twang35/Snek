# b46d-c51atoms201seed1

![b46d-c51atoms201seed1 progress](b46d-c51atoms201seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 485000, avg score 78.13, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | b46d-c51atoms201seed1 |
| seed | 1 |
| zeroed_observations | none |
| learning_rate | 0.0001 |
| adam_epsilon | 0.0003125 |
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
| algo | c51 (distributional), 201 atoms over [-5.0, 120.0] at 0.625 spacing, cross-entropy loss, double (online argmax) target selection, standard init |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | kl (SNEK_PRIORITY_SIGNAL=td_error; a distributional agent has no TD error) |
| importance_sampling_beta | disabled |
| max_steps | 3000000 |
| initial_populate_steps | 1000 |
| eval | 100 episodes every 1000 steps, engine vec, 100 lanes in-process, no worker processes |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.0 |
| CHASE_SAFE_SHAPING | off |
| FREE_SPACE_SHAPING | off |
| eval_only | False |
| min_checkpoint_score | 40.0 |
| c51_support_note | support [-5.0, 120.0] is below the derived maximum return 194.0, so a return above 120.0 would be clipped. 14% headroom over the measured 105.0; spacing 0.625. This is a judgement, not an error. |

## Evals

486 evals so far. Full series in [`b46d-c51atoms201seed1_evals.json`](b46d-c51atoms201seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 1.8 | 1.8 | 0 | 10/95 | 1.3 | 0 | 0.4 |
| 2000 | 1.35 | 1.58 | 0 | 11/95 | 0.85 | 0 | 0.4 |
| ... | | | | | | | |
| 474000 | 68.6 | 60.16 | 5 | 95/95 | 119.03 | 52 | 0.0045 |
| 475000 | 37.33 | 54.15 | 1 | 95/95 | 64.33 | 28 | 0.0045 |
| 476000 | 60.89 | 52.69 | 3 | 95/95 | 107.385 | 48 | 0.0045 |
| 477000 | 64.73 | 51.68 | 5 | 95/95 | 100.415 | 37 | 0.0045 |
| 478000 | 64.21 | 59.15 | 1 | 95/95 | 112.515 | 50 | 0.0044 |
| 479000 | 67.67 | 58.97 | 2 | 95/95 | 109.1 | 43 | 0.0043 |
| 480000 | 71.6 | 65.82 | 3 | 95/95 | 113.89 | 44 | 0.0043 |
| 481000 | 68.0 | 67.24 | 3 | 95/95 | 104.5 | 38 | 0.0043 |
| 482000 | 80.37 | 70.37 | 4 | 95/95 | 135.05 | 56 | 0.0042 |
| 483000 | 61.39 | 69.81 | 1 | 95/95 | 86.765 | 27 | 0.0043 |
| 484000 | 70.63 | 70.4 | 7 | 95/95 | 128.885 | 60 | 0.0043 |
| 485000 | 78.13 | 71.7 | 7 | 95/95 | 136.88 | 60 | 0.0042 |
