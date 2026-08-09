# b20f-fc200seed2

![b20f-fc200seed2 progress](b20f-fc200seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1765000, avg score 91.9, perfect games 50%.

## Config

| setting | value |
|---|---|
| policy_name | b20f-fc200seed2 |
| seed | 2 |
| zeroed_observations | none |
| learning_rate | 1e-05 |
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
| fc_layer_params | (200, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | td_error |
| importance_sampling_beta | 0.4 -> 1.0 over 300000 steps |
| max_steps | 10000000 |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.0 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

1766 evals so far. Full series in [`b20f-fc200seed2_evals.json`](b20f-fc200seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.7 | 0.7 | 0 | 2/95 | 0.2 | 0 | 0.4 |
| 2000 | 0.6 | 0.65 | 0 | 2/95 | 0.1 | 0 | 0.4 |
| ... | | | | | | | |
| 1754000 | 93.3 | 92.28 | 90 | 95/95 | 119.95 | 30 | 0.0051 |
| 1755000 | 91.9 | 91.82 | 81 | 95/95 | 118.55 | 30 | 0.0052 |
| 1756000 | 93.2 | 91.98 | 89 | 95/95 | 119.4 | 30 | 0.0052 |
| 1757000 | 90.6 | 91.9 | 79 | 95/95 | 117.7 | 30 | 0.0053 |
| 1758000 | 92.8 | 92.36 | 87 | 95/95 | 119.0 | 30 | 0.0054 |
| 1759000 | 93.8 | 92.46 | 91 | 95/95 | 140.8 | 50 | 0.0053 |
| 1760000 | 91.3 | 92.34 | 68 | 95/95 | 149.6 | 60 | 0.0052 |
| 1761000 | 90.9 | 91.88 | 79 | 95/95 | 106.7 | 20 | 0.0053 |
| 1762000 | 93.9 | 92.54 | 91 | 95/95 | 140.9 | 50 | 0.0053 |
| 1763000 | 93.4 | 92.66 | 87 | 95/95 | 130.0 | 40 | 0.0054 |
| 1764000 | 93.0 | 92.5 | 82 | 95/95 | 119.65 | 30 | 0.0054 |
| 1765000 | 91.9 | 92.62 | 84 | 95/95 | 139.35 | 50 | 0.0054 |
