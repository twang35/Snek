# b36c-c51fc320seed3

![b36c-c51fc320seed3 progress](b36c-c51fc320seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 183000, avg score 90.0, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | b36c-c51fc320seed3 |
| seed | 3 |
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

184 evals so far. Full series in [`b36c-c51fc320seed3_evals.json`](b36c-c51fc320seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 2.0 | 2.0 | 0 | 11/95 | -3.0 | 0 | 0.4 |
| 1000 | 1.2 | 1.2 | 0 | 11/95 | 0.7 | 0 | 0.4 |
| 2000 | 1.0 | 1.1 | 0 | 5/95 | 0.5 | 0 | 0.4 |
| ... | | | | | | | |
| 172000 | 94.2 | 94.3 | 93 | 95/95 | 152.05 | 60 | 0.0037 |
| 173000 | 93.7 | 94.04 | 84 | 95/95 | 172.8 | 80 | 0.0035 |
| 174000 | 95.0 | 94.08 | 95 | 95/95 | 194.0 | 100 | 0.0033 |
| 175000 | 66.7 | 88.84 | 0 | 95/95 | 135.85 | 70 | 0.0031 |
| 176000 | 87.7 | 87.46 | 26 | 95/95 | 166.8 | 80 | 0.003 |
| 177000 | 81.6 | 84.94 | 27 | 95/95 | 160.25 | 80 | 0.0029 |
| 178000 | 95.0 | 85.2 | 95 | 95/95 | 194.0 | 100 | 0.0027 |
| 179000 | 76.5 | 81.5 | 3 | 95/95 | 144.75 | 70 | 0.0027 |
| 180000 | 94.8 | 87.12 | 93 | 95/95 | 183.85 | 90 | 0.0025 |
| 181000 | 93.6 | 88.3 | 87 | 95/95 | 161.85 | 70 | 0.0024 |
| 182000 | 94.6 | 90.9 | 93 | 95/95 | 173.7 | 80 | 0.0023 |
| 183000 | 90.0 | 89.9 | 76 | 95/95 | 147.85 | 60 | 0.0023 |
