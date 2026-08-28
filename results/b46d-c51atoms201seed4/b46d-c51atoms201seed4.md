# b46d-c51atoms201seed4

![b46d-c51atoms201seed4 progress](b46d-c51atoms201seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 91.73, perfect games 44%.

## Config

| setting | value |
|---|---|
| policy_name | b46d-c51atoms201seed4 |
| seed | 4 |
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

3001 evals so far. Full series in [`b46d-c51atoms201seed4_evals.json`](b46d-c51atoms201seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.07 | 0.07 | 0 | 1/95 | -4.93 | 0 | 0.4 |
| 1000 | 0.79 | 0.79 | 0 | 4/95 | 0.29 | 0 | 0.4 |
| 2000 | 0.55 | 0.67 | 0 | 4/95 | 0.05 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 93.46 | 93.4 | 77 | 95/95 | 147.915 | 56 | 0.0039 |
| 2990000 | 92.15 | 93.06 | 38 | 95/95 | 134.575 | 44 | 0.0039 |
| 2991000 | 92.84 | 92.91 | 77 | 95/95 | 134.86 | 44 | 0.004 |
| 2992000 | 93.42 | 93.01 | 46 | 95/95 | 153.575 | 62 | 0.004 |
| 2993000 | 93.32 | 93.04 | 81 | 95/95 | 134.57 | 43 | 0.004 |
| 2994000 | 94.0 | 93.15 | 83 | 95/95 | 156.505 | 64 | 0.0039 |
| 2995000 | 91.75 | 93.07 | 43 | 95/95 | 115.41 | 26 | 0.004 |
| 2996000 | 91.88 | 92.87 | 39 | 95/95 | 121.78 | 32 | 0.0041 |
| 2997000 | 91.98 | 92.59 | 37 | 95/95 | 132.325 | 42 | 0.0041 |
| 2998000 | 89.85 | 91.89 | 30 | 95/95 | 115.59 | 28 | 0.0042 |
| 2999000 | 92.67 | 91.63 | 47 | 95/95 | 138.805 | 48 | 0.0042 |
| 3000000 | 91.73 | 91.62 | 23 | 95/95 | 133.84 | 44 | 0.0043 |
