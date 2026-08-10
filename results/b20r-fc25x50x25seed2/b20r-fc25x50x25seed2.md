# b20r-fc25x50x25seed2

![b20r-fc25x50x25seed2 progress](b20r-fc25x50x25seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 86.0, perfect games 30%.

## Config

| setting | value |
|---|---|
| policy_name | b20r-fc25x50x25seed2 |
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
| fc_layer_params | (25, 50, 25) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | td_error |
| importance_sampling_beta | 0.4 -> 1.0 over 300000 steps |
| max_steps | 3000000 |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.0 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

3001 evals so far. Full series in [`b20r-fc25x50x25seed2_evals.json`](b20r-fc25x50x25seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.9 | 0.9 | 0 | 3/95 | 0.4 | 0 | 0.4 |
| 2000 | 2.0 | 1.45 | 0 | 8/95 | 1.5 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 93.8 | 93.72 | 91 | 95/95 | 143.05 | 50 | 0.0044 |
| 2990000 | 93.1 | 93.54 | 85 | 95/95 | 141.9 | 50 | 0.0045 |
| 2991000 | 93.7 | 93.7 | 92 | 95/95 | 133.0 | 40 | 0.0044 |
| 2992000 | 94.6 | 93.7 | 93 | 95/95 | 173.7 | 80 | 0.0042 |
| 2993000 | 94.0 | 93.84 | 91 | 95/95 | 152.75 | 60 | 0.0042 |
| 2994000 | 93.0 | 93.68 | 86 | 95/95 | 141.8 | 50 | 0.004 |
| 2995000 | 93.5 | 93.76 | 92 | 95/95 | 122.85 | 30 | 0.0041 |
| 2996000 | 93.7 | 93.76 | 91 | 95/95 | 142.95 | 50 | 0.004 |
| 2997000 | 92.3 | 93.3 | 88 | 95/95 | 111.7 | 20 | 0.0041 |
| 2998000 | 93.6 | 93.22 | 93 | 95/95 | 122.95 | 30 | 0.0042 |
| 2999000 | 92.4 | 93.1 | 81 | 95/95 | 131.25 | 40 | 0.0042 |
| 3000000 | 86.0 | 91.6 | 19 | 95/95 | 115.35 | 30 | 0.0043 |
