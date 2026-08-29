# b35a-chase10g40seed1

![b35a-chase10g40seed1 progress](b35a-chase10g40seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 95.0, perfect games 100%.

## Config

| setting | value |
|---|---|
| policy_name | b35a-chase10g40seed1 |
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
| fc_layer_params | (320,) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | td_error |
| importance_sampling_beta | disabled |
| max_steps | 2000000 |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.0 |
| CHASE_SAFE_SHAPING | c=0.1, potential-based on head/food/tail in one region, gated to snake length >= 40 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2001 evals so far. Full series in [`b35a-chase10g40seed1_evals.json`](b35a-chase10g40seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 2.2 | 2.2 | 0 | 5/95 | 1.7 | 0 | 0.4 |
| 2000 | 1.9 | 2.05 | 0 | 6/95 | 1.4 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 95.0 | 94.28 | 95 | 95/95 | 193.817 | 100 | 0.002 |
| 1990000 | 91.2 | 94.12 | 62 | 95/95 | 160.163 | 70 | 0.002 |
| 1991000 | 92.7 | 93.66 | 72 | 95/95 | 181.569 | 90 | 0.002 |
| 1992000 | 93.9 | 93.56 | 86 | 95/95 | 172.816 | 80 | 0.002 |
| 1993000 | 94.6 | 93.48 | 93 | 95/95 | 173.061 | 80 | 0.002 |
| 1994000 | 94.7 | 93.42 | 92 | 95/95 | 183.575 | 90 | 0.002 |
| 1995000 | 95.0 | 94.18 | 95 | 95/95 | 193.822 | 100 | 0.002 |
| 1996000 | 90.6 | 93.76 | 54 | 95/95 | 169.517 | 80 | 0.002 |
| 1997000 | 94.8 | 93.94 | 93 | 95/95 | 183.683 | 90 | 0.002 |
| 1998000 | 91.9 | 93.4 | 74 | 95/95 | 160.873 | 70 | 0.002 |
| 1999000 | 95.0 | 93.46 | 95 | 95/95 | 193.829 | 100 | 0.002 |
| 2000000 | 95.0 | 93.46 | 95 | 95/95 | 193.809 | 100 | 0.002 |
