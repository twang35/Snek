# b36d-c51fc320seed4

![b36d-c51fc320seed4 progress](b36d-c51fc320seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1873000, avg score 91.3, perfect games 40%.

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

1874 evals so far. Full series in [`b36d-c51fc320seed4_evals.json`](b36d-c51fc320seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 12.8 | 12.8 | 0 | 43/95 | 12.3 | 0 | 0.2 |
| 2000 | 3.1 | 7.95 | 1 | 12/95 | 2.6 | 0 | 0.2 |
| ... | | | | | | | |
| 1862000 | 91.1 | 90.72 | 84 | 95/95 | 100.1 | 10 | 0.0048 |
| 1863000 | 88.6 | 91.82 | 56 | 95/95 | 117.05 | 30 | 0.0049 |
| 1864000 | 86.1 | 90.2 | 16 | 95/95 | 134.45 | 50 | 0.0048 |
| 1865000 | 81.9 | 88.14 | 43 | 95/95 | 109.0 | 30 | 0.0048 |
| 1866000 | 91.2 | 87.78 | 71 | 95/95 | 129.6 | 40 | 0.0047 |
| 1867000 | 93.4 | 88.24 | 87 | 95/95 | 142.65 | 50 | 0.0047 |
| 1868000 | 86.4 | 87.8 | 41 | 95/95 | 114.4 | 30 | 0.0047 |
| 1869000 | 93.0 | 89.18 | 81 | 95/95 | 171.2 | 80 | 0.0046 |
| 1870000 | 93.8 | 91.56 | 91 | 95/95 | 142.6 | 50 | 0.0046 |
| 1871000 | 91.7 | 91.66 | 83 | 95/95 | 119.25 | 30 | 0.0047 |
| 1872000 | 93.4 | 91.66 | 91 | 95/95 | 132.25 | 40 | 0.0047 |
| 1873000 | 91.3 | 92.64 | 69 | 95/95 | 128.8 | 40 | 0.0047 |
