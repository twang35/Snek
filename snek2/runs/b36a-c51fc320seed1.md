# b36a-c51fc320seed1

![b36a-c51fc320seed1 progress](b36a-c51fc320seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 23000, avg score 3.0, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b36a-c51fc320seed1 |
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

24 evals so far. Full series in [`b36a-c51fc320seed1_evals.json`](b36a-c51fc320seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 1.6 | 1.6 | 0 | 4/95 | 1.1 | 0 | 0.4 |
| 2000 | 1.4 | 1.5 | 0 | 7/95 | 0.9 | 0 | 0.4 |
| ... | | | | | | | |
| 12000 | 39.2 | 39.96 | 20 | 58/95 | 34.2 | 0 | 0.0125 |
| 13000 | 57.5 | 42.46 | 42 | 73/95 | 52.5 | 0 | 0.0125 |
| 14000 | 51.8 | 44.92 | 28 | 79/95 | 46.8 | 0 | 0.0125 |
| 15000 | 63.6 | 50.8 | 13 | 95/95 | 69.0 | 10 | 0.0123 |
| 16000 | 69.7 | 56.36 | 30 | 95/95 | 85.5 | 20 | 0.012 |
| 17000 | 18.8 | 52.28 | 2 | 95/95 | 27.8 | 10 | 0.0119 |
| 18000 | 42.3 | 49.24 | 5 | 95/95 | 51.75 | 10 | 0.0118 |
| 19000 | 85.9 | 56.06 | 4 | 95/95 | 174.95 | 90 | 0.0106 |
| 20000 | 3.0 | 43.94 | 0 | 7/95 | 2.5 | 0 | 0.0107 |
| 21000 | 28.1 | 35.62 | 2 | 95/95 | 37.55 | 10 | 0.0107 |
| 22000 | 1.2 | 32.1 | 0 | 6/95 | 0.7 | 0 | 0.0108 |
| 23000 | 3.0 | 24.24 | 0 | 11/95 | 1.15 | 0 | 0.0108 |
