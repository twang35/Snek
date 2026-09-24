# b27n-hist4-seed14

step **100,007,936** · 3052 evals · trailing **94.54** · peak **94.81** @55,607,296 · sef **96.2** · best30 **99.6** @74,743,808

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
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 14 |
| torch_threads | 1 |

![b27n-hist4-seed14](b27n-hist4-seed14.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 1.05 | 1.05 | 0.0 | 11.0 | -0.497 | 0.0 |  |
| 65536 | 2.85 | 16.1 | 0.0 | 56.0 | 2.1 | 0.0 |  |
| 98304 | 22.77 | 11.91 | 0.0 | 47.0 | 18.993 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 93.73 | 94.47 | 21.0 | 95.0 | 190.466 | 98.0 |  |
| 99680256 | 94.65 | 94.48 | 60.0 | 95.0 | 192.416 | 99.0 |  |
| 99713024 | 94.39 | 94.48 | 34.0 | 95.0 | 192.106 | 99.0 |  |
| 99745792 | 94.58 | 94.55 | 53.0 | 95.0 | 192.345 | 99.0 |  |
| 99778560 | 94.16 | 94.54 | 11.0 | 95.0 | 191.878 | 99.0 |  |
| 99811328 | 95.0 | 94.54 | 95.0 | 95.0 | 193.765 | 100.0 |  |
| 99844096 | 94.65 | 94.53 | 60.0 | 95.0 | 192.371 | 99.0 |  |
| 99876864 | 95.0 | 94.54 | 95.0 | 95.0 | 193.759 | 100.0 |  |
| 99909632 | 95.0 | 94.49 | 95.0 | 95.0 | 193.754 | 100.0 |  |
| 99942400 | 93.4 | 94.49 | 8.0 | 95.0 | 190.123 | 98.0 |  |
| 99975168 | 94.23 | 94.52 | 18.0 | 95.0 | 191.992 | 99.0 |  |
| 100007936 | 95.0 | 94.54 | 95.0 | 95.0 | 193.745 | 100.0 |  |
