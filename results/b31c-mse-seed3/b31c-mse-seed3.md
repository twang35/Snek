# b31c-mse-seed3

step **100,007,936** · 3052 evals · trailing **94.57** · peak **94.9** @71,729,152 · sef **97.7** · best30 **99.8** @71,630,848

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
| ppo_discount_final | 0.999 |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | 0.001 |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.95 |
| ppo_gae_lambda_final | 0.999 |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 16.8 |
| ppo_horizon_final | 500.3 |
| ppo_learning_rate | 0.00025 |
| ppo_learning_rate_final | None |
| ppo_minibatch | 512 |
| ppo_normalize_adv | True |
| ppo_rollout | 256 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 32768 |
| ppo_value_loss | mse |
| ppo_vf_coef | 0.5 |
| seed | 3 |
| torch_threads | 1 |

![b31c-mse-seed3](b31c-mse-seed3.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 4.36 | 4.36 | 0.0 | 12.0 | 1.155 | 0.0 |  |
| 65536 | 20.16 | 17.82 | 0.0 | 46.0 | 15.997 | 0.0 |  |
| 98304 | 28.37 | 20.46 | 1.0 | 49.0 | 23.439 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.48 | 95.0 | 95.0 | 193.77 | 100.0 |  |
| 99680256 | 94.12 | 94.47 | 41.0 | 95.0 | 190.858 | 98.0 |  |
| 99713024 | 93.58 | 94.44 | 12.0 | 95.0 | 190.28 | 98.0 |  |
| 99745792 | 94.53 | 94.45 | 48.0 | 95.0 | 192.271 | 99.0 |  |
| 99778560 | 95.0 | 94.47 | 95.0 | 95.0 | 193.776 | 100.0 |  |
| 99811328 | 95.0 | 94.54 | 95.0 | 95.0 | 193.773 | 100.0 |  |
| 99844096 | 95.0 | 94.57 | 95.0 | 95.0 | 193.779 | 100.0 |  |
| 99876864 | 95.0 | 94.47 | 95.0 | 95.0 | 193.779 | 100.0 |  |
| 99909632 | 95.0 | 94.58 | 95.0 | 95.0 | 193.78 | 100.0 |  |
| 99942400 | 94.45 | 94.55 | 40.0 | 95.0 | 192.183 | 99.0 |  |
| 99975168 | 93.32 | 94.55 | 24.0 | 95.0 | 189.061 | 97.0 |  |
| 100007936 | 94.33 | 94.57 | 28.0 | 95.0 | 192.069 | 99.0 |  |
