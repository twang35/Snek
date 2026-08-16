# b32b-c51eps15e4seed2

![b32b-c51eps15e4seed2 progress](b32b-c51eps15e4seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1000000, avg score 78.7, perfect games 30%.

## Config

| setting | value |
|---|---|
| policy_name | b32b-c51eps15e4seed2 |
| seed | 2 |
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

1001 evals so far. Full series in [`b32b-c51eps15e4seed2_evals.json`](b32b-c51eps15e4seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 1/95 | -0.1 | 0 | 0.4 |
| 2000 | 0.7 | 0.55 | 0 | 2/95 | 0.2 | 0 | 0.4 |
| ... | | | | | | | |
| 989000 | 94.8 | 85.36 | 93 | 95/95 | 183.85 | 90 | 0.0031 |
| 990000 | 77.2 | 85.06 | 3 | 95/95 | 155.85 | 80 | 0.003 |
| 991000 | 86.8 | 84.98 | 21 | 95/95 | 165.45 | 80 | 0.003 |
| 992000 | 54.8 | 81.24 | 0 | 95/95 | 82.8 | 30 | 0.003 |
| 993000 | 65.6 | 75.84 | 7 | 95/95 | 114.4 | 50 | 0.003 |
| 994000 | 58.8 | 68.64 | 0 | 95/95 | 107.6 | 50 | 0.003 |
| 995000 | 42.4 | 61.68 | 0 | 95/95 | 70.85 | 30 | 0.0031 |
| 996000 | 67.0 | 57.72 | 14 | 95/95 | 104.05 | 40 | 0.0032 |
| 997000 | 59.7 | 58.7 | 5 | 95/95 | 96.75 | 40 | 0.0033 |
| 998000 | 85.3 | 62.64 | 2 | 95/95 | 153.1 | 70 | 0.0033 |
| 999000 | 56.2 | 62.12 | 0 | 95/95 | 95.5 | 40 | 0.0033 |
| 1000000 | 78.7 | 69.38 | 28 | 95/95 | 107.15 | 30 | 0.0035 |
