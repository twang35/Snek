# b29d-lrclip0-seed4

step **100,007,936** · 3052 evals · trailing **94.54** · peak **94.7** @75,169,792 · sef **93.6** · best30 **99.2** @75,464,704

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
| seed | 4 |
| torch_threads | 1 |

![b29d-lrclip0-seed4](b29d-lrclip0-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 4.22 | 4.22 | 0.0 | 12.0 | 1.778 | 0.0 |  |
| 65536 | 9.19 | 6.71 | 0.0 | 29.0 | 5.446 | 0.0 |  |
| 98304 | 21.34 | 11.58 | 2.0 | 41.0 | 16.477 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.55 | 95.0 | 95.0 | 193.751 | 100.0 |  |
| 99680256 | 94.61 | 94.54 | 56.0 | 95.0 | 192.363 | 99.0 |  |
| 99713024 | 94.07 | 94.52 | 12.0 | 95.0 | 190.825 | 98.0 |  |
| 99745792 | 94.48 | 94.52 | 68.0 | 95.0 | 191.238 | 98.0 |  |
| 99778560 | 95.0 | 94.58 | 95.0 | 95.0 | 193.756 | 100.0 |  |
| 99811328 | 94.75 | 94.52 | 70.0 | 95.0 | 192.5 | 99.0 |  |
| 99844096 | 94.8 | 94.54 | 75.0 | 95.0 | 192.545 | 99.0 |  |
| 99876864 | 94.63 | 94.53 | 58.0 | 95.0 | 192.342 | 99.0 |  |
| 99909632 | 94.39 | 94.53 | 61.0 | 95.0 | 191.091 | 98.0 |  |
| 99942400 | 95.0 | 94.53 | 95.0 | 95.0 | 193.752 | 100.0 |  |
| 99975168 | 93.57 | 94.53 | 30.0 | 95.0 | 189.331 | 97.0 |  |
| 100007936 | 94.46 | 94.54 | 66.0 | 95.0 | 191.143 | 98.0 |  |
