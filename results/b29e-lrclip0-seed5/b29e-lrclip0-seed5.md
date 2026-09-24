# b29e-lrclip0-seed5

step **100,007,936** · 3052 evals · trailing **94.66** · peak **94.81** @98,992,128 · sef **93.4** · best30 **99.5** @99,155,968

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
| ppo_anneal_fraction | 1.0 |
| ppo_clip | 0.2 |
| ppo_clip_final | 0.001 |
| ppo_discount_final | None |
| ppo_entropy_coef | 0.01 |
| ppo_entropy_coef_final | None |
| ppo_epochs | 4 |
| ppo_gae_lambda | 0.95 |
| ppo_gae_lambda_final | None |
| ppo_gradient_clipping | 0.5 |
| ppo_horizon | 16.8 |
| ppo_learning_rate | 0.00025 |
| ppo_learning_rate_final | 0.0 |
| ppo_minibatch | 512 |
| ppo_normalize_adv | True |
| ppo_rollout | 256 |
| ppo_target_kl | 0.0 |
| ppo_transitions_per_rollout | 32768 |
| ppo_value_loss | huber |
| ppo_vf_coef | 0.5 |
| seed | 5 |
| torch_threads | 1 |

![b29e-lrclip0-seed5](b29e-lrclip0-seed5.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 6.52 | 6.52 | 1.0 | 22.0 | 2.532 | 0.0 |  |
| 65536 | 19.07 | 12.79 | 0.0 | 41.0 | 14.612 | 0.0 |  |
| 98304 | 22.15 | 15.91 | 0.0 | 42.0 | 17.345 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 94.75 | 94.76 | 70.0 | 95.0 | 192.512 | 99.0 |  |
| 99680256 | 94.4 | 94.74 | 62.0 | 95.0 | 191.157 | 98.0 |  |
| 99713024 | 94.73 | 94.77 | 68.0 | 95.0 | 192.485 | 99.0 |  |
| 99745792 | 94.06 | 94.78 | 16.0 | 95.0 | 190.831 | 98.0 |  |
| 99778560 | 94.11 | 94.75 | 44.0 | 95.0 | 190.821 | 98.0 |  |
| 99811328 | 95.0 | 94.77 | 95.0 | 95.0 | 193.756 | 100.0 |  |
| 99844096 | 94.64 | 94.64 | 59.0 | 95.0 | 192.392 | 99.0 |  |
| 99876864 | 95.0 | 94.66 | 95.0 | 95.0 | 193.753 | 100.0 |  |
| 99909632 | 94.39 | 94.74 | 57.0 | 95.0 | 191.108 | 98.0 |  |
| 99942400 | 94.69 | 94.73 | 64.0 | 95.0 | 192.449 | 99.0 |  |
| 99975168 | 93.46 | 94.68 | 20.0 | 95.0 | 189.227 | 97.0 |  |
| 100007936 | 94.21 | 94.66 | 16.0 | 95.0 | 191.963 | 99.0 |  |
