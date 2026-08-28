# b46d-c51atoms201seed3

![b46d-c51atoms201seed3 progress](b46d-c51atoms201seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 90.52, perfect games 75%.

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

3001 evals so far. Full series in [`b46d-c51atoms201seed3_evals.json`](b46d-c51atoms201seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 1.03 | 1.03 | 0 | 5/95 | 0.53 | 0 | 0.4 |
| 2000 | 0.69 | 0.86 | 0 | 4/95 | 0.19 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 91.22 | 92.08 | 16 | 95/95 | 144.86 | 55 | 0.0037 |
| 2990000 | 90.99 | 91.97 | 28 | 95/95 | 151.415 | 62 | 0.0037 |
| 2991000 | 91.17 | 91.57 | 28 | 95/95 | 146.845 | 57 | 0.0037 |
| 2992000 | 91.75 | 91.37 | 28 | 95/95 | 153.125 | 63 | 0.0037 |
| 2993000 | 88.07 | 90.64 | 20 | 95/95 | 160.48 | 74 | 0.0036 |
| 2994000 | 88.4 | 90.08 | 16 | 95/95 | 150.5 | 64 | 0.0035 |
| 2995000 | 87.49 | 89.38 | 24 | 95/95 | 147.78 | 62 | 0.0035 |
| 2996000 | 89.91 | 89.12 | 8 | 95/95 | 150.245 | 62 | 0.0035 |
| 2997000 | 84.33 | 87.64 | 19 | 95/95 | 145.39 | 63 | 0.0035 |
| 2998000 | 89.34 | 87.89 | 21 | 95/95 | 144.925 | 57 | 0.0035 |
| 2999000 | 91.26 | 88.47 | 27 | 95/95 | 154.985 | 65 | 0.0034 |
| 3000000 | 90.52 | 89.07 | 17 | 95/95 | 164.195 | 75 | 0.0034 |
