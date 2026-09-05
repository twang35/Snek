# b19n-adameps1e8-seed2

step **376,832** · 23 evals · trailing **52.19** · peak **52.19** @376,832 · sef **0.0** · best30 **0.0** @376,832

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
| ppo_adam_epsilon | 1e-08 |
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
| seed | 2 |
| torch_threads | 1 |

![b19n-adameps1e8-seed2](b19n-adameps1e8-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.58 | 1.58 | 0.0 | 6.0 | -0.669 | 0.0 |  |
| 32768 | 12.29 | 6.93 | 0.0 | 25.0 | 7.387 | 0.0 |  |
| 49152 | 24.37 | 20.82 | 5.0 | 76.0 | 19.325 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 196608 | 56.97 | 30.04 | 24.0 | 87.0 | 52.056 | 0.0 |  |
| 212992 | 57.84 | 44.6 | 30.0 | 82.0 | 53.649 | 0.0 |  |
| 229376 | 62.05 | 48.89 | 7.0 | 88.0 | 57.382 | 0.0 |  |
| 245760 | 61.14 | 50.89 | 30.0 | 89.0 | 57.103 | 0.0 |  |
| 262144 | 65.27 | 33.24 | 26.0 | 89.0 | 61.52 | 0.0 |  |
| 278528 | 66.73 | 48.2 | 28.0 | 92.0 | 63.061 | 0.0 |  |
| 294912 | 75.95 | 38.01 | 35.0 | 95.0 | 78.671 | 5.0 |  |
| 311296 | 79.54 | 40.98 | 51.0 | 95.0 | 79.613 | 2.0 |  |
| 327680 | 80.28 | 46.7 | 49.0 | 95.0 | 84.989 | 6.0 |  |
| 344064 | 80.49 | 50.4 | 49.0 | 95.0 | 85.223 | 6.0 |  |
| 360448 | 82.11 | 43.72 | 58.0 | 95.0 | 84.668 | 4.0 |  |
| 376832 | 80.95 | 52.19 | 37.0 | 95.0 | 83.548 | 4.0 |  |
