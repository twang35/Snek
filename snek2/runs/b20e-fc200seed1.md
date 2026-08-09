# b20e-fc200seed1

![b20e-fc200seed1 progress](b20e-fc200seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1751000, avg score 92.9, perfect games 30%.

## Config

| setting | value |
|---|---|
| policy_name | b20e-fc200seed1 |
| seed | 1 |
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

1752 evals so far. Full series in [`b20e-fc200seed1_evals.json`](b20e-fc200seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.0 | 0 | 0.4 |
| 1000 | 1.3 | 1.3 | 0 | 8/95 | 0.8 | 0 | 0.4 |
| 2000 | 1.2 | 1.25 | 0 | 4/95 | 0.7 | 0 | 0.4 |
| ... | | | | | | | |
| 1740000 | 83.6 | 91.86 | 2 | 95/95 | 100.75 | 20 | 0.0042 |
| 1741000 | 93.8 | 91.74 | 91 | 95/95 | 151.65 | 60 | 0.0042 |
| 1742000 | 92.5 | 91.52 | 89 | 95/95 | 98.35 | 10 | 0.0043 |
| 1743000 | 92.4 | 91.22 | 85 | 95/95 | 130.35 | 40 | 0.0043 |
| 1744000 | 93.2 | 91.1 | 91 | 95/95 | 129.8 | 40 | 0.0043 |
| 1745000 | 94.0 | 93.18 | 93 | 95/95 | 130.6 | 40 | 0.0044 |
| 1746000 | 94.2 | 93.26 | 91 | 95/95 | 162.45 | 70 | 0.0043 |
| 1747000 | 92.4 | 93.24 | 86 | 95/95 | 130.35 | 40 | 0.0044 |
| 1748000 | 94.0 | 93.56 | 91 | 95/95 | 151.4 | 60 | 0.0044 |
| 1749000 | 93.5 | 93.62 | 91 | 95/95 | 130.1 | 40 | 0.0045 |
| 1750000 | 94.1 | 93.64 | 90 | 95/95 | 162.35 | 70 | 0.0043 |
| 1751000 | 92.9 | 93.38 | 91 | 95/95 | 120.0 | 30 | 0.0044 |
