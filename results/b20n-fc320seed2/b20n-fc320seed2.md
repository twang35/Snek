# b20n-fc320seed2

![b20n-fc320seed2 progress](b20n-fc320seed2.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 251000, avg score 91.1, perfect games 10%.

## Config

| setting | value |
|---|---|
| policy_name | b20n-fc320seed2 |
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

252 evals so far. Full series in [`b20n-fc320seed2_evals.json`](b20n-fc320seed2_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -3.2 | 0 | 0.4 |
| 1000 | 0.6 | 0.6 | 0 | 2/95 | 0.1 | 0 | 0.4 |
| 2000 | 0.6 | 0.6 | 0 | 3/95 | 0.1 | 0 | 0.4 |
| ... | | | | | | | |
| 240000 | 93.5 | 92.88 | 90 | 95/95 | 142.75 | 50 | 0.0049 |
| 241000 | 92.8 | 92.78 | 90 | 95/95 | 122.15 | 30 | 0.005 |
| 242000 | 93.1 | 92.74 | 88 | 95/95 | 142.35 | 50 | 0.005 |
| 243000 | 91.9 | 92.88 | 88 | 95/95 | 111.3 | 20 | 0.005 |
| 244000 | 91.7 | 92.6 | 88 | 95/95 | 121.05 | 30 | 0.005 |
| 245000 | 92.4 | 92.38 | 88 | 95/95 | 131.7 | 40 | 0.0049 |
| 246000 | 89.6 | 91.74 | 52 | 95/95 | 128.9 | 40 | 0.0048 |
| 247000 | 91.6 | 91.44 | 88 | 95/95 | 101.05 | 10 | 0.005 |
| 248000 | 92.2 | 91.5 | 84 | 95/95 | 131.5 | 40 | 0.0049 |
| 249000 | 91.7 | 91.5 | 86 | 95/95 | 121.05 | 30 | 0.005 |
| 250000 | 93.3 | 91.68 | 90 | 95/95 | 142.55 | 50 | 0.005 |
| 251000 | 91.1 | 91.98 | 84 | 95/95 | 100.55 | 10 | 0.0052 |
