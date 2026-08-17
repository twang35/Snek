# b36a-c51fc320seed1

![b36a-c51fc320seed1 progress](b36a-c51fc320seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1992000, avg score 93.3, perfect games 60%.

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

1993 evals so far. Full series in [`b36a-c51fc320seed1_evals.json`](b36a-c51fc320seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 1.6 | 1.6 | 0 | 4/95 | 1.1 | 0 | 0.4 |
| 2000 | 1.4 | 1.5 | 0 | 7/95 | 0.9 | 0 | 0.4 |
| ... | | | | | | | |
| 1981000 | 93.7 | 91.78 | 88 | 95/95 | 152.45 | 60 | 0.0046 |
| 1982000 | 93.0 | 91.68 | 87 | 95/95 | 140.9 | 50 | 0.0046 |
| 1983000 | 94.6 | 91.78 | 93 | 95/95 | 173.25 | 80 | 0.0045 |
| 1984000 | 94.4 | 94.14 | 91 | 95/95 | 173.5 | 80 | 0.0044 |
| 1985000 | 91.6 | 93.46 | 67 | 95/95 | 149.9 | 60 | 0.0044 |
| 1986000 | 94.0 | 93.52 | 91 | 95/95 | 162.7 | 70 | 0.0042 |
| 1987000 | 94.4 | 93.8 | 91 | 95/95 | 173.5 | 80 | 0.0041 |
| 1988000 | 90.8 | 93.04 | 58 | 95/95 | 148.65 | 60 | 0.0039 |
| 1989000 | 94.0 | 92.96 | 91 | 95/95 | 152.3 | 60 | 0.0039 |
| 1990000 | 93.9 | 93.42 | 91 | 95/95 | 151.3 | 60 | 0.0038 |
| 1991000 | 91.2 | 92.86 | 75 | 95/95 | 128.25 | 40 | 0.0038 |
| 1992000 | 93.3 | 92.64 | 86 | 95/95 | 152.05 | 60 | 0.0039 |
