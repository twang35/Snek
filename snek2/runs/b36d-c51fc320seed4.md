# b36d-c51fc320seed4

![b36d-c51fc320seed4 progress](b36d-c51fc320seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 21000, avg score 80.7, perfect games 20%.

## Config

| setting | value |
|---|---|
| policy_name | b36d-c51fc320seed4 |
| seed | 4 |
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
| algo | c51 (distributional), 51 atoms over [-5.0, 120.0] at 2.500 spacing, cross-entropy loss, double (online argmax) target selection, standard init |
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
| eval_only | False |
| min_checkpoint_score | 40.0 |
| c51_support_note | support [-5.0, 120.0] is below the derived maximum return 194.0, so a return above 120.0 would be clipped. 14% headroom over the measured 105.0; spacing 2.500. This is a judgement, not an error. |

## Evals

22 evals so far. Full series in [`b36d-c51fc320seed4_evals.json`](b36d-c51fc320seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 12.8 | 12.8 | 0 | 43/95 | 12.3 | 0 | 0.2 |
| 2000 | 3.1 | 7.95 | 1 | 12/95 | 2.6 | 0 | 0.2 |
| ... | | | | | | | |
| 10000 | 17.3 | 6.12 | 2 | 66/95 | 15.9 | 0 | 0.1 |
| 11000 | 26.6 | 10.96 | 1 | 88/95 | 25.2 | 0 | 0.05 |
| 12000 | 38.3 | 18.36 | 1 | 71/95 | 35.1 | 0 | 0.025 |
| 13000 | 43.7 | 26.44 | 12 | 67/95 | 39.15 | 0 | 0.0125 |
| 14000 | 50.8 | 35.34 | 8 | 67/95 | 45.8 | 0 | 0.0125 |
| 15000 | 64.8 | 44.84 | 40 | 87/95 | 59.8 | 0 | 0.0125 |
| 16000 | 56.0 | 50.72 | 23 | 95/95 | 61.4 | 10 | 0.0123 |
| 17000 | 58.6 | 54.78 | 7 | 88/95 | 53.6 | 0 | 0.0123 |
| 18000 | 50.0 | 56.04 | 15 | 83/95 | 45.45 | 0 | 0.0124 |
| 19000 | 24.5 | 50.78 | 3 | 70/95 | 22.2 | 0 | 0.0124 |
| 20000 | 72.8 | 52.38 | 3 | 94/95 | 68.25 | 0 | 0.0124 |
| 21000 | 80.7 | 57.32 | 63 | 95/95 | 96.5 | 20 | 0.0121 |
