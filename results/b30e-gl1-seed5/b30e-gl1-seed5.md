# b30e-gl1-seed5

step **100,007,936** · 3052 evals · trailing **94.68** · peak **94.93** @84,934,656 · sef **95.1** · best30 **99.9** @84,934,656

## Config

| | |
|---|---|
| algo | ppo |
| collect_envs | 128 |
| discount | 0.99 |
| eval_interval | 32768 |
| eval_queue | True |
| eval_queue_depth | 16 |
| eval_workers | 8 |
| fc_layers | (320,) |
| graph_eval_episodes | 100 |
| max_steps | 100007936 |
| min_checkpoint_score | 40.0 |
| ppo_adam_epsilon | 1e-07 |
| ppo_anneal_fraction | 0.5 |
| ppo_clip | 0.2 |
| ppo_clip_final | None |
| ppo_discount_final | 1.0 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | 0.001 |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.95 |
| ppo_gae_lambda_final | 1.0 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 16.8 |
| ppo_horizon_final | inf |
| ppo_learning_rate | 0.00025 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 512 |
| ppo_normalize_adv | True |
| ppo_rollout | 256 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 32768 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 5 |
| torch_threads | 1 |

![b30e-gl1-seed5](b30e-gl1-seed5.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 6.84 | 6.84 | 1.0 | 19.0 | 2.404 | 0.0 |  |
| 65536 | 20.7 | 13.77 | 4.0 | 41.0 | 15.749 | 0.0 |  |
| 98304 | 24.6 | 17.38 | 7.0 | 42.0 | 19.565 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.72 | 95.0 | 95.0 | 193.768 | 100.0 |  |
| 99680256 | 95.0 | 94.64 | 95.0 | 95.0 | 193.761 | 100.0 |  |
| 99713024 | 94.58 | 94.68 | 53.0 | 95.0 | 192.298 | 99.0 |  |
| 99745792 | 95.0 | 94.74 | 95.0 | 95.0 | 193.759 | 100.0 |  |
| 99778560 | 94.47 | 94.75 | 42.0 | 95.0 | 192.24 | 99.0 |  |
| 99811328 | 93.41 | 94.72 | 18.0 | 95.0 | 189.097 | 97.0 |  |
| 99844096 | 94.63 | 94.71 | 58.0 | 95.0 | 192.393 | 99.0 |  |
| 99876864 | 94.68 | 94.73 | 63.0 | 95.0 | 192.4 | 99.0 |  |
| 99909632 | 94.58 | 94.74 | 67.0 | 95.0 | 191.302 | 98.0 |  |
| 99942400 | 95.0 | 94.76 | 95.0 | 95.0 | 193.758 | 100.0 |  |
| 99975168 | 95.0 | 94.78 | 95.0 | 95.0 | 193.763 | 100.0 |  |
| 100007936 | 93.78 | 94.68 | 14.0 | 95.0 | 189.51 | 97.0 |  |
