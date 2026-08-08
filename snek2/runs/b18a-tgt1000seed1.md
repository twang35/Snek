# b18a-tgt1000seed1

![b18a-tgt1000seed1 progress](b18a-tgt1000seed1.png)

Blue is average score (food eaten) on the left axis, red is perfect-game percentage on the right.

Latest eval: step 2612000, avg score 93.6, perfect games 50%.

## Config

| setting | value |
|---|---|
| policy_name | b18a-tgt1000seed1 |
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
| FOOD_DISTANCE_REWARD | 0.0 |
| eval_only | False |
| min_checkpoint_score | 40.0 |

## Evals

2613 evals so far. Full series in [`b18a-tgt1000seed1_evals.json`](b18a-tgt1000seed1_evals.json).

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0 | 0/95 | -5.0 | 0 | 0.4 |
| 1000 | 0.0 | 0.0 | 0 | 0/95 | -0.5 | 0 | 0.4 |
| 2000 | 0.0 | 0.0 | 0 | 0/95 | -0.5 | 0 | 0.4 |
| ... | | | | | | | |
| 2601000 | 94.6 | 94.58 | 91 | 95/95 | 183.2 | 90 | 0.0027 |
| 2602000 | 93.5 | 94.32 | 82 | 95/95 | 171.7 | 80 | 0.0027 |
| 2603000 | 94.5 | 94.26 | 93 | 95/95 | 162.3 | 70 | 0.0027 |
| 2604000 | 93.0 | 94.0 | 82 | 95/95 | 161.7 | 70 | 0.0027 |
| 2605000 | 93.6 | 93.84 | 91 | 95/95 | 130.2 | 40 | 0.0027 |
| 2606000 | 94.0 | 93.72 | 91 | 95/95 | 151.85 | 60 | 0.0027 |
| 2607000 | 93.9 | 93.8 | 93 | 95/95 | 130.5 | 40 | 0.0027 |
| 2608000 | 93.4 | 93.58 | 91 | 95/95 | 120.05 | 30 | 0.0028 |
| 2609000 | 93.6 | 93.7 | 91 | 95/95 | 130.2 | 40 | 0.0029 |
| 2610000 | 94.8 | 93.94 | 93 | 95/95 | 183.4 | 90 | 0.0028 |
| 2611000 | 94.0 | 93.94 | 91 | 95/95 | 152.3 | 60 | 0.0029 |
| 2612000 | 93.6 | 93.88 | 91 | 95/95 | 140.6 | 50 | 0.0029 |
