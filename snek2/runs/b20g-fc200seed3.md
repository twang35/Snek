# b20g-fc200seed3

![b20g-fc200seed3 progress](b20g-fc200seed3.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 1732000, avg score 93.8, perfect games 50%.

## Config

| setting | value |
|---|---|
| policy_name | b20g-fc200seed3 |
| seed | 3 |
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

1733 evals so far. Full series in [`b20g-fc200seed3_evals.json`](b20g-fc200seed3_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.8 | 0.8 | 0 | 4/95 | -4.2 | 0 | 0.4 |
| 1000 | 0.8 | 0.8 | 0 | 4/95 | 0.3 | 0 | 0.4 |
| 2000 | 1.0 | 0.9 | 0 | 2/95 | 0.5 | 0 | 0.4 |
| ... | | | | | | | |
| 1721000 | 93.9 | 92.88 | 90 | 95/95 | 151.75 | 60 | 0.003 |
| 1722000 | 94.0 | 94.24 | 93 | 95/95 | 141.0 | 50 | 0.0031 |
| 1723000 | 94.0 | 94.08 | 93 | 95/95 | 141.45 | 50 | 0.0031 |
| 1724000 | 93.3 | 93.96 | 88 | 95/95 | 131.25 | 40 | 0.0032 |
| 1725000 | 89.6 | 92.96 | 58 | 95/95 | 147.9 | 60 | 0.0033 |
| 1726000 | 93.2 | 92.82 | 89 | 95/95 | 140.2 | 50 | 0.0033 |
| 1727000 | 86.8 | 91.38 | 30 | 95/95 | 124.75 | 40 | 0.0032 |
| 1728000 | 93.8 | 91.34 | 91 | 95/95 | 151.65 | 60 | 0.0032 |
| 1729000 | 94.0 | 91.48 | 91 | 95/95 | 162.7 | 70 | 0.0032 |
| 1730000 | 94.8 | 92.52 | 93 | 95/95 | 183.4 | 90 | 0.0031 |
| 1731000 | 93.3 | 92.54 | 91 | 95/95 | 130.8 | 40 | 0.0032 |
| 1732000 | 93.8 | 93.94 | 91 | 95/95 | 141.25 | 50 | 0.0032 |
