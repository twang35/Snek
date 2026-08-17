# b36c-c51fc320seed3

![b36c-c51fc320seed3 progress](b36c-c51fc320seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1980000, avg score 94.2, perfect games 70%.

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

1981 evals so far. Full series in [`b36c-c51fc320seed3_evals.json`](b36c-c51fc320seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 2.0 | 2.0 | 0 | 11/95 | -3.0 | 0 | 0.4 |
| 1000 | 1.2 | 1.2 | 0 | 11/95 | 0.7 | 0 | 0.4 |
| 2000 | 1.0 | 1.1 | 0 | 5/95 | 0.5 | 0 | 0.4 |
| ... | | | | | | | |
| 1969000 | 88.0 | 91.52 | 27 | 95/95 | 166.65 | 80 | 0.0028 |
| 1970000 | 89.0 | 90.54 | 63 | 95/95 | 106.6 | 20 | 0.0029 |
| 1971000 | 93.0 | 91.06 | 77 | 95/95 | 171.65 | 80 | 0.0029 |
| 1972000 | 94.6 | 91.36 | 93 | 95/95 | 173.25 | 80 | 0.0028 |
| 1973000 | 85.4 | 90.0 | 35 | 95/95 | 103.9 | 20 | 0.003 |
| 1974000 | 87.3 | 89.86 | 25 | 95/95 | 145.6 | 60 | 0.003 |
| 1975000 | 85.4 | 89.14 | 47 | 95/95 | 143.7 | 60 | 0.003 |
| 1976000 | 85.3 | 87.6 | 55 | 95/95 | 113.3 | 30 | 0.003 |
| 1977000 | 94.0 | 87.48 | 93 | 95/95 | 143.25 | 50 | 0.003 |
| 1978000 | 82.9 | 86.98 | 45 | 95/95 | 130.8 | 50 | 0.003 |
| 1979000 | 77.1 | 84.94 | 20 | 95/95 | 115.05 | 40 | 0.0031 |
| 1980000 | 94.2 | 86.7 | 91 | 95/95 | 162.9 | 70 | 0.0031 |
