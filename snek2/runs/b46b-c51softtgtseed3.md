# b46b-c51softtgtseed3

![b46b-c51softtgtseed3 progress](b46b-c51softtgtseed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 94.49, perfect games 76%.

Training was resumed at step 856000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | b46b-c51softtgtseed3 |
| seed | 3 |
| zeroed_observations | none |
| learning_rate | 0.0001 |
| adam_epsilon | 0.0003125 |
| perfect_game_reward | 100.0 |
| batch_size | 128 |
| discount | 0.9975 |
| target_update_period | 1 |
| target_update_tau | 0.005 |
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
| eval | 100 episodes every 1000 steps, engine vec, 100 lanes in-process, no worker processes |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.0 |
| CHASE_SAFE_SHAPING | off |
| FREE_SPACE_SHAPING | off |
| eval_only | False |
| min_checkpoint_score | 40.0 |
| c51_support_note | support [-5.0, 120.0] is below the derived maximum return 194.0, so a return above 120.0 would be clipped. 14% headroom over the measured 105.0; spacing 2.500. This is a judgement, not an error. |

## Evals

3001 evals so far. Full series in [`b46b-c51softtgtseed3_evals.json`](b46b-c51softtgtseed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 1.2 | 1.2 | 0 | 11/95 | -3.8 | 0 | 0.4 |
| 1000 | 1.85 | 1.85 | 0 | 11/95 | 1.35 | 0 | 0.4 |
| 2000 | 1.5 | 1.68 | 0 | 3/95 | 1.0 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 89.55 | 91.45 | 29 | 95/95 | 114.795 | 28 | 0.0028 |
| 2990000 | 92.78 | 91.5 | 33 | 95/95 | 161.57 | 70 | 0.0028 |
| 2991000 | 90.35 | 91.63 | 25 | 95/95 | 138.25 | 50 | 0.0028 |
| 2992000 | 93.76 | 91.74 | 76 | 95/95 | 156.085 | 64 | 0.0028 |
| 2993000 | 90.57 | 91.4 | 47 | 95/95 | 124.495 | 36 | 0.0028 |
| 2994000 | 93.44 | 92.18 | 76 | 95/95 | 145.77 | 54 | 0.0029 |
| 2995000 | 92.29 | 92.08 | 16 | 95/95 | 159.815 | 69 | 0.0029 |
| 2996000 | 94.32 | 92.88 | 88 | 95/95 | 165.825 | 73 | 0.0029 |
| 2997000 | 92.38 | 92.6 | 22 | 95/95 | 153.53 | 63 | 0.0029 |
| 2998000 | 92.88 | 93.06 | 31 | 95/95 | 157.15 | 66 | 0.0029 |
| 2999000 | 93.11 | 93.0 | 20 | 95/95 | 153.625 | 62 | 0.0029 |
| 3000000 | 94.49 | 93.44 | 91 | 95/95 | 169.025 | 76 | 0.0029 |
