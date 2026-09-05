# b19o-adameps1e8-seed3

step **393,216** · 23 evals · trailing **45.87** · peak **49.5** @360,448 · sef **0.0** · best30 **0.0** @393,216

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
| seed | 3 |
| torch_threads | 1 |

![b19o-adameps1e8-seed3](b19o-adameps1e8-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.0 | 0.0 | 0.0 | 0.0 | -4.112 | 0.0 |  |
| 32768 | 2.19 | 1.09 | 1.0 | 10.0 | 1.575 | 0.0 |  |
| 49152 | 12.47 | 11.96 | 0.0 | 42.0 | 9.461 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 196608 | 56.32 | 46.39 | 4.0 | 95.0 | 53.074 | 1.0 |  |
| 212992 | 61.92 | 31.71 | 12.0 | 95.0 | 61.804 | 3.0 |  |
| 229376 | 59.86 | 44.31 | 21.0 | 95.0 | 56.749 | 1.0 |  |
| 245760 | 58.91 | 48.32 | 3.0 | 91.0 | 55.205 | 0.0 |  |
| 278528 | 78.09 | 43.4 | 11.0 | 95.0 | 84.876 | 8.0 |  |
| 294912 | 76.35 | 47.82 | 1.0 | 95.0 | 84.153 | 9.0 |  |
| 311296 | 74.73 | 28.96 | 19.0 | 95.0 | 78.565 | 5.0 |  |
| 327680 | 76.17 | 35.13 | 1.0 | 95.0 | 77.982 | 3.0 |  |
| 344064 | 73.58 | 41.23 | 13.0 | 92.0 | 72.415 | 0.0 |  |
| 360448 | 75.41 | 49.5 | 22.0 | 95.0 | 76.229 | 2.0 |  |
| 376832 | 77.54 | 39.07 | 5.0 | 95.0 | 79.347 | 3.0 |  |
| 393216 | 73.88 | 45.87 | 34.0 | 92.0 | 72.729 | 0.0 |  |
