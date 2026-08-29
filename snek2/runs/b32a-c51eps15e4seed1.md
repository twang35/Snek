# b32a-c51eps15e4seed1

![b32a-c51eps15e4seed1 progress](b32a-c51eps15e4seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1000000, avg score 92.1, perfect games 50%.

## Config

| setting | value |
|---|---|
| policy_name | b32a-c51eps15e4seed1 |
| seed | 1 |
| zeroed_observations | none |
| learning_rate | 0.0001 |
| adam_epsilon | 0.00015 |
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
| algo | c51 (distributional), 51 atoms over [-5.0, 120.0] at 2.500 spacing, cross-entropy loss, double (online argmax) target selection, standard init |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | kl (SNEK_PRIORITY_SIGNAL=td_error; a distributional agent has no TD error) |
| importance_sampling_beta | disabled |
| max_steps | 1000000 |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.0 |
| CHASE_SAFE_SHAPING | off |
| eval_only | False |
| min_checkpoint_score | 40.0 |
| c51_support_note | support [-5.0, 120.0] is below the derived maximum return 194.0, so a return above 120.0 would be clipped. Measured max is 105.0 (14% headroom); spacing 2.500. This is a judgement, not an error. |

## Evals

1001 evals so far. Full series in [`b32a-c51eps15e4seed1_evals.json`](b32a-c51eps15e4seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 0.8 | 0.8 | 0 | 3/95 | 0.3 | 0 | 0.4 |
| 2000 | 0.8 | 0.8 | 0 | 4/95 | 0.3 | 0 | 0.4 |
| ... | | | | | | | |
| 989000 | 87.8 | 88.84 | 43 | 95/95 | 124.85 | 40 | 0.0033 |
| 990000 | 90.1 | 89.12 | 57 | 95/95 | 138.45 | 50 | 0.0034 |
| 991000 | 88.1 | 88.2 | 34 | 95/95 | 146.4 | 60 | 0.0034 |
| 992000 | 93.6 | 89.46 | 87 | 95/95 | 151.45 | 60 | 0.0033 |
| 993000 | 85.1 | 88.94 | 16 | 95/95 | 102.7 | 20 | 0.0035 |
| 994000 | 94.2 | 90.22 | 91 | 95/95 | 162.45 | 70 | 0.0035 |
| 995000 | 87.4 | 89.68 | 26 | 95/95 | 155.65 | 70 | 0.0035 |
| 996000 | 92.0 | 90.46 | 75 | 95/95 | 139.9 | 50 | 0.0035 |
| 997000 | 86.2 | 88.98 | 24 | 95/95 | 124.15 | 40 | 0.0035 |
| 998000 | 70.7 | 86.1 | 23 | 95/95 | 118.6 | 50 | 0.0034 |
| 999000 | 80.6 | 83.38 | 11 | 95/95 | 138.0 | 60 | 0.0035 |
| 1000000 | 92.1 | 84.32 | 82 | 95/95 | 140.9 | 50 | 0.0035 |
