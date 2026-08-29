# b46d-c51atoms201seed4

![b46d-c51atoms201seed4 progress](b46d-c51atoms201seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 578000, avg score 92.55, perfect games 63%.

## Config

| setting | value |
|---|---|
| policy_name | b46d-c51atoms201seed4 |
| seed | 4 |
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

579 evals so far. Full series in [`b46d-c51atoms201seed4_evals.json`](b46d-c51atoms201seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.07 | 0.07 | 0 | 1/95 | -4.93 | 0 | 0.4 |
| 1000 | 0.79 | 0.79 | 0 | 4/95 | 0.29 | 0 | 0.4 |
| 2000 | 0.55 | 0.67 | 0 | 4/95 | 0.05 | 0 | 0.4 |
| ... | | | | | | | |
| 567000 | 91.59 | 92.85 | 27 | 95/95 | 168.34 | 78 | 0.0024 |
| 568000 | 91.93 | 92.7 | 16 | 95/95 | 161.49 | 71 | 0.0024 |
| 569000 | 93.26 | 92.71 | 22 | 95/95 | 178.06 | 86 | 0.0024 |
| 570000 | 92.56 | 92.68 | 10 | 95/95 | 170.35 | 79 | 0.0023 |
| 571000 | 91.35 | 92.14 | 0 | 95/95 | 167.285 | 77 | 0.0024 |
| 572000 | 93.93 | 92.61 | 63 | 95/95 | 168.87 | 76 | 0.0023 |
| 573000 | 89.21 | 92.06 | 12 | 95/95 | 158.77 | 71 | 0.0024 |
| 574000 | 89.97 | 91.4 | 1 | 95/95 | 152.385 | 64 | 0.0024 |
| 575000 | 92.21 | 91.33 | 20 | 95/95 | 173.98 | 83 | 0.0024 |
| 576000 | 87.32 | 90.53 | 1 | 95/95 | 149.96 | 64 | 0.0024 |
| 577000 | 91.61 | 90.06 | 4 | 95/95 | 162.435 | 72 | 0.0024 |
| 578000 | 92.55 | 90.73 | 31 | 95/95 | 154.105 | 63 | 0.0024 |
