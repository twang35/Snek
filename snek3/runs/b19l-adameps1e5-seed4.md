# b19l-adameps1e5-seed4

step **393,216** · 24 evals · trailing **54.31** · peak **54.31** @393,216 · sef **0.0** · best30 **0.0** @393,216

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 16384 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| graph_eval_episodes | 100 |
| max_steps | 50003968 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-05 |
| ppo_anneal_fraction | 1.0 |
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.98 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 33.6 |
| ppo_learning_rate | 0.0003 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 256 |
| ppo_normalize_adv | True |
| ppo_rollout | 128 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 16384 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 4 |
| torch_threads | 1 |

![b19l-adameps1e5-seed4](b19l-adameps1e5-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.24 | 0.24 | 0.0 | 2.0 | -0.576 | 0.0 |  |
| 32768 | 10.3 | 19.36 | 0.0 | 28.0 | 7.048 | 0.0 |  |
| 49152 | 26.65 | 18.0 | 5.0 | 46.0 | 21.607 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 212992 | 51.31 | 32.13 | 9.0 | 92.0 | 46.233 | 0.0 |  |
| 229376 | 61.2 | 50.41 | 16.0 | 95.0 | 58.784 | 2.0 |  |
| 245760 | 74.35 | 52.89 | 20.0 | 95.0 | 74.02 | 3.0 |  |
| 262144 | 77.25 | 42.08 | 40.0 | 95.0 | 82.187 | 8.0 |  |
| 278528 | 76.09 | 48.02 | 6.0 | 95.0 | 87.482 | 14.0 |  |
| 294912 | 78.13 | 36.61 | 26.0 | 95.0 | 79.809 | 4.0 |  |
| 311296 | 83.39 | 39.73 | 32.0 | 95.0 | 88.552 | 7.0 |  |
| 327680 | 81.35 | 44.39 | 20.0 | 95.0 | 89.973 | 10.0 |  |
| 344064 | 83.42 | 51.91 | 32.0 | 95.0 | 92.1 | 10.0 |  |
| 360448 | 81.79 | 46.46 | 7.0 | 95.0 | 88.436 | 8.0 |  |
| 376832 | 85.06 | 49.88 | 56.0 | 95.0 | 100.666 | 17.0 |  |
| 393216 | 87.05 | 54.31 | 37.0 | 95.0 | 103.671 | 18.0 |  |
