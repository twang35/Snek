# b31f-mse-seed6

step **100,007,936** · 3052 evals · trailing **94.57** · peak **94.91** @83,296,256 · sef **97.9** · best30 **99.9** @85,426,176

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
| seed | 6 |
| torch_threads | 1 |

![b31f-mse-seed6](b31f-mse-seed6.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 4.49 | 4.49 | 0.0 | 15.0 | 1.15 | 0.0 |  |
| 65536 | 16.58 | 10.54 | 1.0 | 46.0 | 13.292 | 0.0 |  |
| 98304 | 24.34 | 15.14 | 2.0 | 47.0 | 19.608 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.46 | 95.0 | 95.0 | 193.766 | 100.0 |  |
| 99680256 | 95.0 | 94.53 | 95.0 | 95.0 | 193.775 | 100.0 |  |
| 99713024 | 93.7 | 94.53 | 8.0 | 95.0 | 190.448 | 98.0 |  |
| 99745792 | 94.63 | 94.55 | 58.0 | 95.0 | 192.407 | 99.0 |  |
| 99778560 | 94.99 | 94.55 | 94.0 | 95.0 | 192.71 | 99.0 |  |
| 99811328 | 95.0 | 94.56 | 95.0 | 95.0 | 193.774 | 100.0 |  |
| 99844096 | 95.0 | 94.56 | 95.0 | 95.0 | 193.776 | 100.0 |  |
| 99876864 | 94.54 | 94.54 | 61.0 | 95.0 | 191.275 | 98.0 |  |
| 99909632 | 95.0 | 94.57 | 95.0 | 95.0 | 193.768 | 100.0 |  |
| 99942400 | 94.37 | 94.57 | 32.0 | 95.0 | 192.102 | 99.0 |  |
| 99975168 | 95.0 | 94.57 | 95.0 | 95.0 | 193.769 | 100.0 |  |
| 100007936 | 95.0 | 94.57 | 95.0 | 95.0 | 193.774 | 100.0 |  |
