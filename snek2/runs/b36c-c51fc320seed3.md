# b36c-c51fc320seed3

![b36c-c51fc320seed3 progress](b36c-c51fc320seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2011000, avg score 93.9, perfect games 50%.

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

2012 evals so far. Full series in [`b36c-c51fc320seed3_evals.json`](b36c-c51fc320seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 2.0 | 2.0 | 0 | 11/95 | -3.0 | 0 | 0.4 |
| 1000 | 1.2 | 1.2 | 0 | 11/95 | 0.7 | 0 | 0.4 |
| 2000 | 1.0 | 1.1 | 0 | 5/95 | 0.5 | 0 | 0.4 |
| ... | | | | | | | |
| 2000000 | 94.7 | 93.18 | 92 | 95/95 | 183.3 | 90 | 0.003 |
| 2001000 | 91.6 | 93.34 | 76 | 95/95 | 139.95 | 50 | 0.0031 |
| 2002000 | 91.3 | 93.08 | 60 | 95/95 | 169.95 | 80 | 0.0031 |
| 2003000 | 94.6 | 93.4 | 93 | 95/95 | 173.7 | 80 | 0.003 |
| 2004000 | 93.3 | 93.1 | 91 | 95/95 | 121.75 | 30 | 0.003 |
| 2005000 | 90.4 | 92.24 | 51 | 95/95 | 169.05 | 80 | 0.003 |
| 2006000 | 86.9 | 91.3 | 31 | 95/95 | 145.2 | 60 | 0.0029 |
| 2007000 | 91.5 | 91.34 | 80 | 95/95 | 120.4 | 30 | 0.003 |
| 2008000 | 91.5 | 90.72 | 80 | 95/95 | 119.5 | 30 | 0.003 |
| 2009000 | 94.0 | 90.86 | 91 | 95/95 | 153.2 | 60 | 0.003 |
| 2010000 | 94.8 | 91.74 | 93 | 95/95 | 183.4 | 90 | 0.0029 |
| 2011000 | 93.9 | 93.14 | 92 | 95/95 | 142.25 | 50 | 0.0029 |
