# b46d-c51atoms201seed2

![b46d-c51atoms201seed2 progress](b46d-c51atoms201seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 575000, avg score 86.8, perfect games 47%.

## Config

| setting | value |
|---|---|
| policy_name | b46d-c51atoms201seed2 |
| seed | 2 |
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

576 evals so far. Full series in [`b46d-c51atoms201seed2_evals.json`](b46d-c51atoms201seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.04 | 0.04 | 0 | 1/95 | -4.96 | 0 | 0.4 |
| 1000 | 0.79 | 0.79 | 0 | 10/95 | 0.29 | 0 | 0.4 |
| 2000 | 1.32 | 1.06 | 0 | 6/95 | 0.82 | 0 | 0.4 |
| ... | | | | | | | |
| 564000 | 90.54 | 90.79 | 64 | 95/95 | 154.58 | 65 | 0.0022 |
| 565000 | 89.25 | 90.3 | 51 | 95/95 | 151.255 | 63 | 0.0022 |
| 566000 | 90.51 | 89.62 | 7 | 95/95 | 157.58 | 68 | 0.0022 |
| 567000 | 90.48 | 89.84 | 67 | 95/95 | 161.395 | 72 | 0.0022 |
| 568000 | 92.03 | 90.56 | 66 | 95/95 | 166.97 | 76 | 0.0022 |
| 569000 | 89.47 | 90.35 | 23 | 95/95 | 152.56 | 64 | 0.0023 |
| 570000 | 93.44 | 91.19 | 21 | 95/95 | 180.365 | 88 | 0.0022 |
| 571000 | 91.07 | 91.3 | 72 | 95/95 | 150.815 | 61 | 0.0023 |
| 572000 | 91.58 | 91.52 | 39 | 95/95 | 159.24 | 69 | 0.0023 |
| 573000 | 88.46 | 90.8 | 21 | 95/95 | 132.515 | 46 | 0.0023 |
| 574000 | 89.26 | 90.76 | 54 | 95/95 | 149.955 | 62 | 0.0024 |
| 575000 | 86.8 | 89.43 | 12 | 95/95 | 132.39 | 47 | 0.0024 |
