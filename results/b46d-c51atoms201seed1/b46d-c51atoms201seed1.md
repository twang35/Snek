# b46d-c51atoms201seed1

![b46d-c51atoms201seed1 progress](b46d-c51atoms201seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 84.09, perfect games 58%.

## Config

| setting | value |
|---|---|
| policy_name | b46d-c51atoms201seed1 |
| seed | 1 |
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

3001 evals so far. Full series in [`b46d-c51atoms201seed1_evals.json`](b46d-c51atoms201seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 1.8 | 1.8 | 0 | 10/95 | 1.3 | 0 | 0.4 |
| 2000 | 1.35 | 1.58 | 0 | 11/95 | 0.85 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 80.64 | 83.78 | 14 | 95/95 | 127.365 | 49 | 0.0038 |
| 2990000 | 93.15 | 85.65 | 18 | 95/95 | 165.06 | 73 | 0.0037 |
| 2991000 | 81.16 | 84.7 | 14 | 95/95 | 144.165 | 65 | 0.0037 |
| 2992000 | 79.41 | 84.39 | 9 | 95/95 | 127.4 | 50 | 0.0037 |
| 2993000 | 73.04 | 81.48 | 5 | 95/95 | 111.58 | 41 | 0.0037 |
| 2994000 | 78.89 | 81.13 | 12 | 95/95 | 121.68 | 45 | 0.0037 |
| 2995000 | 49.81 | 72.46 | 12 | 95/95 | 65.11 | 19 | 0.0038 |
| 2996000 | 50.67 | 66.36 | 10 | 95/95 | 65.02 | 18 | 0.0039 |
| 2997000 | 48.87 | 60.26 | 10 | 95/95 | 64.035 | 19 | 0.004 |
| 2998000 | 38.07 | 53.26 | 3 | 95/95 | 44.78 | 11 | 0.0041 |
| 2999000 | 58.44 | 49.17 | 11 | 95/95 | 75.23 | 20 | 0.0042 |
| 3000000 | 84.09 | 56.03 | 1 | 95/95 | 140.265 | 58 | 0.0041 |
