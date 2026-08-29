# b36a-c51fc320seed1

![b36a-c51fc320seed1 progress](b36a-c51fc320seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2023000, avg score 90.6, perfect games 10%.

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

2024 evals so far. Full series in [`b36a-c51fc320seed1_evals.json`](b36a-c51fc320seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 1.6 | 1.6 | 0 | 4/95 | 1.1 | 0 | 0.4 |
| 2000 | 1.4 | 1.5 | 0 | 7/95 | 0.9 | 0 | 0.4 |
| ... | | | | | | | |
| 2012000 | 93.1 | 93.06 | 91 | 95/95 | 121.1 | 30 | 0.0037 |
| 2013000 | 93.4 | 93.0 | 91 | 95/95 | 130.9 | 40 | 0.0038 |
| 2014000 | 93.5 | 92.9 | 89 | 95/95 | 151.35 | 60 | 0.0039 |
| 2015000 | 94.0 | 93.18 | 91 | 95/95 | 151.85 | 60 | 0.0039 |
| 2016000 | 92.5 | 93.3 | 85 | 95/95 | 120.5 | 30 | 0.004 |
| 2017000 | 92.5 | 93.18 | 88 | 95/95 | 120.5 | 30 | 0.0042 |
| 2018000 | 93.6 | 93.22 | 89 | 95/95 | 151.45 | 60 | 0.0042 |
| 2019000 | 92.6 | 93.04 | 83 | 95/95 | 129.65 | 40 | 0.0042 |
| 2020000 | 92.7 | 92.78 | 86 | 95/95 | 141.05 | 50 | 0.0043 |
| 2021000 | 93.8 | 93.04 | 91 | 95/95 | 141.7 | 50 | 0.0042 |
| 2022000 | 90.4 | 92.62 | 79 | 95/95 | 107.55 | 20 | 0.0044 |
| 2023000 | 90.6 | 92.02 | 87 | 95/95 | 96.9 | 10 | 0.0044 |
