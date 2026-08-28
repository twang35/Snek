# b46d-c51atoms201seed3

![b46d-c51atoms201seed3 progress](b46d-c51atoms201seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 509000, avg score 69.61, perfect games 55%.

## Config

| setting | value |
|---|---|
| policy_name | b46d-c51atoms201seed3 |
| seed | 3 |
| zeroed_observations | none |
| learning_rate | 0.0001 |
| adam_epsilon | 0.0003125 |
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
| algo | c51 (distributional), 201 atoms over [-5.0, 120.0] at 0.625 spacing, cross-entropy loss, double (online argmax) target selection, standard init |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | kl (SNEK_PRIORITY_SIGNAL=td_error; a distributional agent has no TD error) |
| importance_sampling_beta | disabled |
| max_steps | 3000000 |
| initial_populate_steps | 1000 |
| eval | 100 episodes every 1000 steps, engine vec, 100 lanes in-process, no worker processes |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.0 |
| CHASE_SAFE_SHAPING | off |
| FREE_SPACE_SHAPING | off |
| eval_only | False |
| min_checkpoint_score | 40.0 |
| c51_support_note | support [-5.0, 120.0] is below the derived maximum return 194.0, so a return above 120.0 would be clipped. 14% headroom over the measured 105.0; spacing 0.625. This is a judgement, not an error. |

## Evals

510 evals so far. Full series in [`b46d-c51atoms201seed3_evals.json`](b46d-c51atoms201seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 1.03 | 1.03 | 0 | 5/95 | 0.53 | 0 | 0.4 |
| 2000 | 0.69 | 0.86 | 0 | 4/95 | 0.19 | 0 | 0.4 |
| ... | | | | | | | |
| 498000 | 87.1 | 88.02 | 12 | 95/95 | 161.725 | 76 | 0.0027 |
| 499000 | 83.35 | 86.69 | 8 | 95/95 | 145.175 | 63 | 0.0027 |
| 500000 | 83.29 | 85.45 | 8 | 95/95 | 149.955 | 68 | 0.0027 |
| 501000 | 82.45 | 85.04 | 9 | 95/95 | 139.935 | 59 | 0.0027 |
| 502000 | 83.61 | 83.96 | 10 | 95/95 | 150.955 | 69 | 0.0027 |
| 503000 | 87.39 | 84.02 | 9 | 95/95 | 162.24 | 76 | 0.0027 |
| 504000 | 68.41 | 81.03 | 5 | 95/95 | 119.115 | 53 | 0.0027 |
| 505000 | 73.95 | 79.16 | 6 | 95/95 | 122.89 | 51 | 0.0027 |
| 506000 | 81.15 | 78.9 | 2 | 95/95 | 146.55 | 67 | 0.0027 |
| 507000 | 73.69 | 76.92 | 2 | 95/95 | 124.395 | 53 | 0.0027 |
| 508000 | 73.87 | 74.21 | 8 | 95/95 | 120.775 | 49 | 0.0028 |
| 509000 | 69.61 | 74.45 | 3 | 95/95 | 122.26 | 55 | 0.0028 |
