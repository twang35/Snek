# b27o-hist4-seed15

step **100,007,936** · 3052 evals · trailing **94.63** · peak **94.93** @59,965,440 · sef **95.9** · best30 **99.9** @60,162,048

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
| seed | 15 |
| torch_threads | 1 |

![b27o-hist4-seed15](b27o-hist4-seed15.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 6.29 | 6.29 | 0.0 | 22.0 | 4.083 | 0.0 |  |
| 65536 | 24.86 | 20.08 | 3.0 | 42.0 | 20.493 | 0.0 |  |
| 98304 | 32.66 | 25.1 | 9.0 | 60.0 | 27.842 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 93.7 | 94.55 | 18.0 | 95.0 | 190.438 | 98.0 |  |
| 99680256 | 95.0 | 94.54 | 95.0 | 95.0 | 193.763 | 100.0 |  |
| 99713024 | 94.16 | 94.52 | 11.0 | 95.0 | 191.891 | 99.0 |  |
| 99745792 | 95.0 | 94.57 | 95.0 | 95.0 | 193.752 | 100.0 |  |
| 99778560 | 95.0 | 94.57 | 95.0 | 95.0 | 193.767 | 100.0 |  |
| 99811328 | 95.0 | 94.55 | 95.0 | 95.0 | 193.761 | 100.0 |  |
| 99844096 | 94.59 | 94.54 | 54.0 | 95.0 | 192.357 | 99.0 |  |
| 99876864 | 95.0 | 94.6 | 95.0 | 95.0 | 193.766 | 100.0 |  |
| 99909632 | 93.95 | 94.56 | 20.0 | 95.0 | 190.722 | 98.0 |  |
| 99942400 | 95.0 | 94.6 | 95.0 | 95.0 | 193.762 | 100.0 |  |
| 99975168 | 94.52 | 94.61 | 47.0 | 95.0 | 192.286 | 99.0 |  |
| 100007936 | 95.0 | 94.63 | 95.0 | 95.0 | 193.77 | 100.0 |  |
