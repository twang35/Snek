# b20l-fc200x50seed4

![b20l-fc200x50seed4 progress](b20l-fc200x50seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 740000, avg score 93.6, perfect games 40%.

## Config

| setting | value |
|---|---|
| policy_name | b20l-fc200x50seed4 |
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
| fc_layer_params | (200, 50) |
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

741 evals so far. Full series in [`b20l-fc200x50seed4_evals.json`](b20l-fc200x50seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.1 | 0.1 | 0 | 1/95 | -4.9 | 0 | 0.4 |
| 1000 | 0.2 | 0.2 | 0 | 1/95 | -0.3 | 0 | 0.4 |
| 2000 | 0.4 | 0.3 | 0 | 2/95 | -0.1 | 0 | 0.4 |
| ... | | | | | | | |
| 729000 | 93.6 | 93.32 | 90 | 95/95 | 162.75 | 70 | 0.0054 |
| 730000 | 93.3 | 93.34 | 88 | 95/95 | 142.55 | 50 | 0.0052 |
| 731000 | 91.6 | 92.92 | 88 | 95/95 | 120.95 | 30 | 0.0053 |
| 732000 | 93.3 | 93.06 | 91 | 95/95 | 132.6 | 40 | 0.0054 |
| 733000 | 94.1 | 93.18 | 90 | 95/95 | 163.25 | 70 | 0.0053 |
| 734000 | 92.7 | 93.0 | 91 | 95/95 | 122.05 | 30 | 0.0053 |
| 735000 | 91.8 | 92.7 | 86 | 95/95 | 111.2 | 20 | 0.0053 |
| 736000 | 92.9 | 92.96 | 86 | 95/95 | 132.2 | 40 | 0.0052 |
| 737000 | 93.3 | 92.96 | 82 | 95/95 | 162.45 | 70 | 0.005 |
| 738000 | 93.9 | 92.92 | 90 | 95/95 | 153.1 | 60 | 0.0049 |
| 739000 | 93.2 | 93.02 | 88 | 95/95 | 132.5 | 40 | 0.0049 |
| 740000 | 93.6 | 93.38 | 92 | 95/95 | 132.9 | 40 | 0.0048 |
