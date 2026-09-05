# b19j-adameps1e5-seed2

step **393,216** · 24 evals · trailing **49.86** · peak **52.65** @245,760 · sef **0.0** · best30 **0.0** @393,216

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
| seed | 2 |
| torch_threads | 1 |

![b19j-adameps1e5-seed2](b19j-adameps1e5-seed2.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 1.58 | 1.58 | 0.0 | 5.0 | -0.578 | 0.0 |  |
| 32768 | 10.67 | 6.12 | 0.0 | 22.0 | 6.676 | 0.0 |  |
| 49152 | 19.41 | 10.55 | 7.0 | 44.0 | 14.513 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 212992 | 67.62 | 34.98 | 29.0 | 95.0 | 65.624 | 2.0 |  |
| 229376 | 72.63 | 51.9 | 29.0 | 95.0 | 70.955 | 2.0 |  |
| 245760 | 69.93 | 52.65 | 19.0 | 95.0 | 70.655 | 4.0 |  |
| 262144 | 71.8 | 39.41 | 17.0 | 95.0 | 71.699 | 3.0 |  |
| 278528 | 70.64 | 47.4 | 25.0 | 95.0 | 72.958 | 5.0 |  |
| 294912 | 70.93 | 41.51 | 15.0 | 95.0 | 72.258 | 4.0 |  |
| 311296 | 73.26 | 32.01 | 8.0 | 95.0 | 75.035 | 3.0 |  |
| 327680 | 74.15 | 45.33 | 43.0 | 95.0 | 75.912 | 3.0 |  |
| 344064 | 73.37 | 48.7 | 3.0 | 95.0 | 75.107 | 3.0 |  |
| 360448 | 73.88 | 43.53 | 5.0 | 95.0 | 75.72 | 3.0 |  |
| 376832 | 74.05 | 50.96 | 10.0 | 95.0 | 79.899 | 7.0 |  |
| 393216 | 73.05 | 49.86 | 33.0 | 92.0 | 71.952 | 0.0 |  |
