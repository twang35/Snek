# b33d-c51win10seed4

![b33d-c51win10seed4 progress](b33d-c51win10seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1708000, avg score 59.7, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b33d-c51win10seed4 |
| seed | 4 |
| zeroed_observations | none |
| learning_rate | 0.0001 |
| adam_epsilon | 0.00015 |
| perfect_game_reward | 10.0 |
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
| fc_layer_params | (200, 100, 100) |
| algo | c51 (distributional), 51 atoms over [-5.0, 40.0] at 0.900 spacing, cross-entropy loss, double (online argmax) target selection, standard init |
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
| c51_support_note | support [-5.0, 40.0] is below the derived maximum return 104.0, so a return above 40.0 would be clipped. 21% headroom over the measured 33.0; spacing 0.900. This is a judgement, not an error. |

## Evals

1709 evals so far. Full series in [`b33d-c51win10seed4_evals.json`](b33d-c51win10seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.3 | 0.3 | 0 | 1/95 | -4.7 | 0 | 0.4 |
| 1000 | 1.4 | 1.4 | 0 | 4/95 | 0.9 | 0 | 0.4 |
| 2000 | 4.2 | 2.8 | 1 | 13/95 | 3.7 | 0 | 0.4 |
| ... | | | | | | | |
| 1697000 | 51.6 | 69.36 | 22 | 93/95 | 47.05 | 0 | 0.0111 |
| 1698000 | 68.8 | 68.0 | 28 | 93/95 | 64.7 | 0 | 0.0111 |
| 1699000 | 64.6 | 65.46 | 16 | 93/95 | 60.5 | 0 | 0.0112 |
| 1700000 | 68.0 | 63.28 | 4 | 95/95 | 64.85 | 10 | 0.0111 |
| 1701000 | 73.9 | 65.38 | 30 | 95/95 | 72.15 | 20 | 0.011 |
| 1702000 | 78.3 | 70.72 | 23 | 95/95 | 76.95 | 10 | 0.0111 |
| 1703000 | 48.7 | 66.7 | 11 | 95/95 | 45.1 | 10 | 0.0111 |
| 1704000 | 58.1 | 65.4 | 18 | 95/95 | 54.95 | 10 | 0.011 |
| 1705000 | 74.4 | 66.68 | 22 | 95/95 | 73.6 | 30 | 0.0108 |
| 1706000 | 79.9 | 67.88 | 29 | 95/95 | 81.9 | 50 | 0.0104 |
| 1707000 | 69.8 | 66.18 | 17 | 93/95 | 65.25 | 0 | 0.0104 |
| 1708000 | 59.7 | 68.38 | 15 | 93/95 | 54.7 | 0 | 0.0106 |
