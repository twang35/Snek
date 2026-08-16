# b33c-c51win10seed3

![b33c-c51win10seed3 progress](b33c-c51win10seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1772000, avg score 82.3, perfect games 0%.

## Config

| setting | value |
|---|---|
| policy_name | b33c-c51win10seed3 |
| seed | 3 |
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

1773 evals so far. Full series in [`b33c-c51win10seed3_evals.json`](b33c-c51win10seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 1/95 | -0.1 | 0 | 0.4 |
| 2000 | 0.3 | 0.35 | 0 | 1/95 | -0.2 | 0 | 0.4 |
| ... | | | | | | | |
| 1761000 | 79.2 | 70.58 | 36 | 92/95 | 75.1 | 0 | 0.0118 |
| 1762000 | 68.6 | 70.38 | 22 | 95/95 | 65.45 | 10 | 0.0117 |
| 1763000 | 67.6 | 69.1 | 28 | 90/95 | 63.5 | 0 | 0.0117 |
| 1764000 | 74.9 | 71.4 | 39 | 94/95 | 70.35 | 0 | 0.0117 |
| 1765000 | 74.1 | 72.88 | 43 | 92/95 | 70.45 | 0 | 0.0117 |
| 1766000 | 75.1 | 72.06 | 39 | 95/95 | 72.4 | 10 | 0.0116 |
| 1767000 | 75.9 | 73.52 | 38 | 94/95 | 72.25 | 0 | 0.0116 |
| 1768000 | 68.1 | 73.62 | 35 | 93/95 | 63.55 | 0 | 0.0116 |
| 1769000 | 86.4 | 75.92 | 46 | 95/95 | 84.15 | 10 | 0.0115 |
| 1770000 | 85.0 | 78.1 | 48 | 94/95 | 80.45 | 0 | 0.0115 |
| 1771000 | 60.0 | 75.08 | 32 | 95/95 | 56.4 | 10 | 0.0114 |
| 1772000 | 82.3 | 76.36 | 47 | 93/95 | 79.55 | 0 | 0.0114 |
