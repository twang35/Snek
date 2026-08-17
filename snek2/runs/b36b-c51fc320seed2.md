# b36b-c51fc320seed2

![b36b-c51fc320seed2 progress](b36b-c51fc320seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1972000, avg score 94.3, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b36b-c51fc320seed2 |
| seed | 2 |
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

1973 evals so far. Full series in [`b36b-c51fc320seed2_evals.json`](b36b-c51fc320seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.2 | 0.2 | 0 | 1/95 | -0.3 | 0 | 0.4 |
| 2000 | 0.4 | 0.3 | 0 | 2/95 | -0.1 | 0 | 0.4 |
| ... | | | | | | | |
| 1961000 | 93.8 | 94.06 | 93 | 95/95 | 132.2 | 40 | 0.0037 |
| 1962000 | 90.0 | 93.22 | 61 | 95/95 | 138.8 | 50 | 0.0037 |
| 1963000 | 91.2 | 92.58 | 67 | 95/95 | 130.5 | 40 | 0.0037 |
| 1964000 | 93.2 | 92.4 | 91 | 95/95 | 131.6 | 40 | 0.0038 |
| 1965000 | 94.0 | 92.44 | 91 | 95/95 | 153.2 | 60 | 0.0038 |
| 1966000 | 93.5 | 92.38 | 86 | 95/95 | 152.7 | 60 | 0.0038 |
| 1967000 | 94.4 | 93.26 | 93 | 95/95 | 163.55 | 70 | 0.0038 |
| 1968000 | 90.2 | 93.06 | 75 | 95/95 | 139.0 | 50 | 0.0037 |
| 1969000 | 93.6 | 93.14 | 89 | 95/95 | 141.5 | 50 | 0.0036 |
| 1970000 | 90.1 | 92.36 | 56 | 95/95 | 129.4 | 40 | 0.0037 |
| 1971000 | 87.5 | 91.16 | 24 | 95/95 | 156.65 | 70 | 0.0036 |
| 1972000 | 94.3 | 91.14 | 88 | 95/95 | 183.35 | 90 | 0.0035 |
