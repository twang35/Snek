# b46c-c51atoms21seed1

![b46c-c51atoms21seed1 progress](b46c-c51atoms21seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 92.47, perfect games 60%.

## Config

| setting | value |
|---|---|
| policy_name | b46c-c51atoms21seed1 |
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
| algo | c51 (distributional), 21 atoms over [-5.0, 120.0] at 6.250 spacing, cross-entropy loss, double (online argmax) target selection, standard init |
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
| c51_support_note | support [-5.0, 120.0] is below the derived maximum return 194.0, so a return above 120.0 would be clipped. 14% headroom over the measured 105.0; spacing 6.250. This is a judgement, not an error. |

## Evals

3001 evals so far. Full series in [`b46c-c51atoms21seed1_evals.json`](b46c-c51atoms21seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.81 | 0.81 | 0 | 8/95 | 0.31 | 0 | 0.4 |
| 2000 | 0.78 | 0.8 | 0 | 5/95 | 0.28 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 93.78 | 93.6 | 68 | 95/95 | 154.385 | 62 | 0.0031 |
| 2990000 | 90.99 | 93.09 | 43 | 95/95 | 116.775 | 28 | 0.0032 |
| 2991000 | 93.01 | 92.96 | 22 | 95/95 | 159.585 | 68 | 0.0031 |
| 2992000 | 93.09 | 92.89 | 25 | 95/95 | 145.645 | 54 | 0.0031 |
| 2993000 | 93.98 | 92.97 | 82 | 95/95 | 155.715 | 63 | 0.0032 |
| 2994000 | 91.85 | 92.58 | 72 | 95/95 | 123.015 | 33 | 0.0032 |
| 2995000 | 93.99 | 93.18 | 89 | 95/95 | 153.735 | 61 | 0.0032 |
| 2996000 | 91.47 | 92.88 | 28 | 95/95 | 138.645 | 49 | 0.0033 |
| 2997000 | 89.45 | 92.15 | 12 | 95/95 | 123.42 | 36 | 0.0034 |
| 2998000 | 89.65 | 91.28 | 22 | 95/95 | 138.77 | 51 | 0.0034 |
| 2999000 | 92.38 | 91.39 | 31 | 95/95 | 141.77 | 51 | 0.0034 |
| 3000000 | 92.47 | 91.08 | 24 | 95/95 | 151.085 | 60 | 0.0034 |
