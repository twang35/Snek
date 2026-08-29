# b1c-baseline-seed3

step **3,000,000** · 3000 evals · trailing **92.87** · peak **93.41** @2,874,000 · sef **0.0** · best30 **56.7** @2,975,000

## Config

| | |
|---|---|
| adam_epsilon | 1e-07 |
| algo | dqn |
| batch_size | 128 |
| beta_anneal_steps | 300000 |
| collect_envs | 1 |
| discount | 0.99 |
| eval_interval | 1000 |
| fc_layers | (320,) |
| fork_branches | 4 |
| fork_max_steps | 60 |
| fork_min_length | 85 |
| fork_prob | 0.5 |
| gradient_clipping | 0.0 |
| graph_eval_episodes | 100 |
| guided_fraction | 0.8 |
| initial_collect_steps | 2000 |
| initial_epsilon | 0.4 |
| is_beta | 0.4 |
| is_beta_final | 1.0 |
| is_weights | True |
| learning_rate | 1e-05 |
| max_steps | 3000000 |
| min_checkpoint_score | 40.0 |
| min_epsilon | 0.002 |
| n_step_update | 1 |
| priority_exponent | 0.6 |
| replay_buffer_max_length | 100000 |
| replay_ratio | 1.0 |
| seed | 3 |
| target_update_period | 8 |
| target_update_tau | 1.0 |
| torch_threads | 1 |

![b1c-baseline-seed3](b1c-baseline-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 1000 | 0.75 | 0.75 | 0.0 | 6.0 | 0.197 | 0.0 | 0.4 |
| 2000 | 4.07 | 2.41 | 0.0 | 20.0 | 3.503 | 0.0 | 0.4 |
| 3000 | 6.0 | 3.61 | 1.0 | 20.0 | 5.425 | 0.0 | 0.2 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 2989000 | 91.94 | 93.04 | 73.0 | 95.0 | 124.958 | 35.0 | 0.00376 |
| 2990000 | 92.44 | 93.0 | 49.0 | 95.0 | 136.591 | 46.0 | 0.00381 |
| 2991000 | 92.08 | 92.95 | 45.0 | 95.0 | 131.96 | 42.0 | 0.00387 |
| 2992000 | 93.0 | 92.93 | 77.0 | 95.0 | 135.133 | 44.0 | 0.00391 |
| 2993000 | 93.23 | 92.92 | 77.0 | 95.0 | 141.337 | 50.0 | 0.00393 |
| 2994000 | 92.96 | 92.91 | 71.0 | 95.0 | 137.899 | 47.0 | 0.00396 |
| 2995000 | 93.19 | 92.91 | 53.0 | 95.0 | 144.047 | 53.0 | 0.004 |
| 2996000 | 92.57 | 92.87 | 75.0 | 95.0 | 131.296 | 41.0 | 0.00405 |
| 2997000 | 92.36 | 92.89 | 67.0 | 95.0 | 129.456 | 39.0 | 0.00409 |
| 2998000 | 92.56 | 92.86 | 61.0 | 95.0 | 129.091 | 39.0 | 0.00414 |
| 2999000 | 93.32 | 92.88 | 81.0 | 95.0 | 141.22 | 50.0 | 0.00416 |
| 3000000 | 92.94 | 92.87 | 71.0 | 95.0 | 140.709 | 50.0 | 0.00418 |
