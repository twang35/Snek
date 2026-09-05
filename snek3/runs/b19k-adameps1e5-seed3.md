# b19k-adameps1e5-seed3

step **376,832** · 23 evals · trailing **49.11** · peak **49.51** @245,760 · sef **0.0** · best30 **0.0** @376,832

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
| seed | 3 |
| torch_threads | 1 |

![b19k-adameps1e5-seed3](b19k-adameps1e5-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 16384 | 0.1 | 0.1 | 0.0 | 1.0 | -3.522 | 0.0 |  |
| 32768 | 6.35 | 3.22 | 0.0 | 21.0 | 4.831 | 0.0 |  |
| 49152 | 18.25 | 8.23 | 0.0 | 35.0 | 13.675 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 196608 | 51.01 | 46.32 | 4.0 | 84.0 | 46.012 | 0.0 |  |
| 212992 | 58.15 | 33.29 | 22.0 | 95.0 | 54.379 | 1.0 |  |
| 229376 | 61.62 | 42.83 | 17.0 | 92.0 | 57.595 | 0.0 |  |
| 245760 | 58.45 | 49.51 | 19.0 | 92.0 | 53.841 | 0.0 |  |
| 262144 | 62.77 | 39.42 | 5.0 | 95.0 | 59.668 | 1.0 |  |
| 278528 | 69.64 | 44.32 | 16.0 | 94.0 | 66.205 | 0.0 |  |
| 294912 | 67.21 | 29.51 | 21.0 | 90.0 | 63.716 | 0.0 |  |
| 311296 | 73.49 | 36.38 | 30.0 | 95.0 | 72.103 | 1.0 |  |
| 327680 | 75.18 | 41.66 | 20.0 | 95.0 | 75.006 | 1.0 |  |
| 344064 | 76.53 | 47.76 | 24.0 | 95.0 | 78.236 | 3.0 |  |
| 360448 | 77.68 | 46.07 | 32.0 | 95.0 | 77.331 | 1.0 |  |
| 376832 | 77.43 | 49.11 | 3.0 | 95.0 | 78.23 | 2.0 |  |
