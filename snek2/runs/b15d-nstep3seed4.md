# b15d-nstep3seed4

![b15d-nstep3seed4 progress](b15d-nstep3seed4.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 5807000, avg score 94.9, perfect games 90%.

## Config

| setting | value |
|---|---|
| policy_name | b15d-nstep3seed4 |
| seed | 4 |
| zeroed_observations | none |
| learning_rate | 1e-05 |
| batch_size | 128 |
| discount | 0.995 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| gradient_clipping | none |
| n_step_update | 3 |
| initial_epsilon | 0.4 |
| min_epsilon | 0.002 |
| epsilon_schedule | bootstrap on avg_reward [2, 5, 10, 15, 20] then geometric to floor by 80% trailing-30 perfect |
| guided_fraction | 0.8 |
| exploration_shield | 80% of refinement-phase episodes draw the epsilon move from non-fatal actions; greedy moves never shielded |
| fc_layer_params | (50, 100, 50) |
| replay_buffer | cpprb prioritized, capacity 100000 |
| priority_exponent (alpha) | 0.6 |
| priority_signal | td_loss |
| importance_sampling_beta | disabled |
| max_steps | 10000000 |
| initial_populate_steps | 1000 |
| eval | 10 episodes every 1000 steps |
| grid | 9x9, max possible score 95 |
| DEATH_REWARD | -5.0 |
| FOOD_REWARD | 1.0 |
| FOOD_DISTANCE_REWARD | 0.001 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

5808 evals so far. Full series in [`b15d-nstep3seed4_evals.json`](b15d-nstep3seed4_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.5 | 0.5 | 0 | 1/95 | -4.503 | 0 | 0.4 |
| 1000 | 0.2 | 0.2 | 0 | 1/95 | -4.816 | 0 | 0.4 |
| 2000 | 0.2 | 0.2 | 0 | 1/95 | -4.801 | 0 | 0.4 |
| ... | | | | | | | |
| 5796000 | 94.7 | 94.46 | 92 | 95/95 | 183.142 | 90 | 0.002 |
| 5797000 | 94.2 | 94.42 | 90 | 95/95 | 172.709 | 80 | 0.002 |
| 5798000 | 95.0 | 94.58 | 95 | 95/95 | 193.455 | 100 | 0.002 |
| 5799000 | 95.0 | 94.7 | 95 | 95/95 | 193.437 | 100 | 0.002 |
| 5800000 | 92.2 | 94.22 | 82 | 95/95 | 150.213 | 60 | 0.002 |
| 5801000 | 94.0 | 94.08 | 91 | 95/95 | 162.011 | 70 | 0.002 |
| 5802000 | 94.3 | 94.1 | 88 | 95/95 | 182.735 | 90 | 0.002 |
| 5803000 | 95.0 | 94.1 | 95 | 95/95 | 193.48 | 100 | 0.002 |
| 5804000 | 93.5 | 93.8 | 88 | 95/95 | 161.499 | 70 | 0.002 |
| 5805000 | 93.8 | 94.12 | 90 | 95/95 | 151.485 | 60 | 0.002 |
| 5806000 | 93.9 | 94.1 | 88 | 95/95 | 161.565 | 70 | 0.002 |
| 5807000 | 94.9 | 94.22 | 94 | 95/95 | 182.926 | 90 | 0.002 |
