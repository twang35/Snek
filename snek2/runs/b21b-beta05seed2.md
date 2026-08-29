# b21b-beta05seed2

![b21b-beta05seed2 progress](b21b-beta05seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 3000000, avg score 94.1, perfect games 60%.

Training was resumed at step 164000 (the dashed lines on the graph).

## Config

| setting | value |
|---|---|
| policy_name | b21b-beta05seed2 |
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
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | td_error |
| importance_sampling_beta | 0.4 -> 0.5 over 300000 steps |
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

3001 evals so far. Full series in [`b21b-beta05seed2_evals.json`](b21b-beta05seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.9 | 0.9 | 0 | 3/95 | 0.4 | 0 | 0.4 |
| 2000 | 1.0 | 0.95 | 0 | 3/95 | 0.5 | 0 | 0.4 |
| ... | | | | | | | |
| 2989000 | 94.5 | 93.82 | 92 | 95/95 | 173.15 | 80 | 0.0028 |
| 2990000 | 93.9 | 93.92 | 92 | 95/95 | 140.9 | 50 | 0.0028 |
| 2991000 | 93.4 | 93.98 | 84 | 95/95 | 161.65 | 70 | 0.0029 |
| 2992000 | 92.9 | 93.8 | 88 | 95/95 | 140.8 | 50 | 0.0029 |
| 2993000 | 93.7 | 93.68 | 90 | 95/95 | 152.45 | 60 | 0.0029 |
| 2994000 | 94.4 | 93.66 | 91 | 95/95 | 173.05 | 80 | 0.0029 |
| 2995000 | 94.2 | 93.72 | 92 | 95/95 | 162.9 | 70 | 0.0029 |
| 2996000 | 93.9 | 93.82 | 88 | 95/95 | 162.15 | 70 | 0.0029 |
| 2997000 | 95.0 | 94.24 | 95 | 95/95 | 194.0 | 100 | 0.0028 |
| 2998000 | 94.7 | 94.44 | 92 | 95/95 | 183.75 | 90 | 0.0027 |
| 2999000 | 94.5 | 94.46 | 92 | 95/95 | 173.15 | 80 | 0.0027 |
| 3000000 | 94.1 | 94.44 | 92 | 95/95 | 152.4 | 60 | 0.0028 |
