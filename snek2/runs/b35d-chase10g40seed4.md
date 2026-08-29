# b35d-chase10g40seed4

![b35d-chase10g40seed4 progress](b35d-chase10g40seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 94.8, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b35d-chase10g40seed4 |
| seed | 4 |
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

2001 evals so far. Full series in [`b35d-chase10g40seed4_evals.json`](b35d-chase10g40seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 0.4 | 0.4 | 0 | 2/95 | -0.1 | 0 | 0.4 |
| 2000 | 2.1 | 1.25 | 1 | 6/95 | 1.6 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 93.2 | 94.32 | 77 | 95/95 | 182.082 | 90 | 0.002 |
| 1990000 | 92.4 | 94.12 | 69 | 95/95 | 181.27 | 90 | 0.002 |
| 1991000 | 91.9 | 93.5 | 77 | 95/95 | 170.82 | 80 | 0.002 |
| 1992000 | 95.0 | 93.5 | 95 | 95/95 | 193.816 | 100 | 0.002 |
| 1993000 | 95.0 | 93.5 | 95 | 95/95 | 193.827 | 100 | 0.002 |
| 1994000 | 93.8 | 93.62 | 83 | 95/95 | 182.673 | 90 | 0.002 |
| 1995000 | 95.0 | 94.14 | 95 | 95/95 | 193.821 | 100 | 0.002 |
| 1996000 | 93.7 | 94.5 | 85 | 95/95 | 172.602 | 80 | 0.002 |
| 1997000 | 93.5 | 94.2 | 83 | 95/95 | 172.414 | 80 | 0.002 |
| 1998000 | 94.8 | 94.16 | 93 | 95/95 | 183.678 | 90 | 0.002 |
| 1999000 | 94.8 | 94.36 | 93 | 95/95 | 183.672 | 90 | 0.002 |
| 2000000 | 94.8 | 94.32 | 93 | 95/95 | 183.659 | 90 | 0.002 |
