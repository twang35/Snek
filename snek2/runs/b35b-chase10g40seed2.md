# b35b-chase10g40seed2

![b35b-chase10g40seed2 progress](b35b-chase10g40seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2000000, avg score 91.6, perfect games 80%.

## Config

| setting | value |
|---|---|
| policy_name | b35b-chase10g40seed2 |
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

2001 evals so far. Full series in [`b35b-chase10g40seed2_evals.json`](b35b-chase10g40seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -3.2 | 0 | 0.4 |
| 1000 | 0.6 | 0.6 | 0 | 2/95 | 0.1 | 0 | 0.4 |
| 2000 | 0.6 | 0.6 | 0 | 3/95 | 0.1 | 0 | 0.4 |
| ... | | | | | | | |
| 1989000 | 95.0 | 93.08 | 95 | 95/95 | 193.798 | 100 | 0.002 |
| 1990000 | 92.2 | 92.52 | 72 | 95/95 | 171.079 | 80 | 0.002 |
| 1991000 | 95.0 | 92.52 | 95 | 95/95 | 193.804 | 100 | 0.002 |
| 1992000 | 93.9 | 92.34 | 84 | 95/95 | 182.747 | 90 | 0.002 |
| 1993000 | 95.0 | 94.22 | 95 | 95/95 | 193.775 | 100 | 0.002 |
| 1994000 | 85.6 | 92.34 | 1 | 95/95 | 174.474 | 90 | 0.002 |
| 1995000 | 95.0 | 92.9 | 95 | 95/95 | 193.805 | 100 | 0.002 |
| 1996000 | 94.5 | 92.8 | 90 | 95/95 | 183.348 | 90 | 0.002 |
| 1997000 | 94.9 | 93.0 | 94 | 95/95 | 183.29 | 90 | 0.002 |
| 1998000 | 92.3 | 92.46 | 68 | 95/95 | 181.144 | 90 | 0.002 |
| 1999000 | 83.5 | 92.04 | 1 | 95/95 | 162.419 | 80 | 0.002 |
| 2000000 | 91.6 | 91.36 | 72 | 95/95 | 170.508 | 80 | 0.002 |
