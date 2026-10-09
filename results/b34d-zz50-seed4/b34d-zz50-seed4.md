# b34d-zz50-seed4

step **100,007,936** · 3052 evals · trailing **94.85** · peak **94.94** @82,280,448 · sef **95.4** · best30 **99.9** @82,673,664

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
| init_from | None |
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
| seed | 4 |
| torch_threads | 1 |

![b34d-zz50-seed4](b34d-zz50-seed4.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 3.11 | 3.11 | 0.0 | 10.0 | 1.055 | 0.0 |  |
| 65536 | 12.49 | 12.0 | 0.0 | 30.0 | 8.337 | 0.0 |  |
| 98304 | 20.39 | 11.75 | 2.0 | 44.0 | 15.403 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.86 | 95.0 | 95.0 | 193.766 | 100.0 |  |
| 99680256 | 95.0 | 94.84 | 95.0 | 95.0 | 193.768 | 100.0 |  |
| 99713024 | 95.0 | 94.86 | 95.0 | 95.0 | 193.781 | 100.0 |  |
| 99745792 | 95.0 | 94.86 | 95.0 | 95.0 | 193.77 | 100.0 |  |
| 99778560 | 94.95 | 94.82 | 90.0 | 95.0 | 192.731 | 99.0 |  |
| 99811328 | 94.65 | 94.84 | 60.0 | 95.0 | 192.431 | 99.0 |  |
| 99844096 | 94.39 | 94.82 | 34.0 | 95.0 | 192.164 | 99.0 |  |
| 99876864 | 95.0 | 94.84 | 95.0 | 95.0 | 193.772 | 100.0 |  |
| 99909632 | 95.0 | 94.84 | 95.0 | 95.0 | 193.772 | 100.0 |  |
| 99942400 | 95.0 | 94.85 | 95.0 | 95.0 | 193.771 | 100.0 |  |
| 99975168 | 95.0 | 94.87 | 95.0 | 95.0 | 193.776 | 100.0 |  |
| 100007936 | 95.0 | 94.85 | 95.0 | 95.0 | 193.777 | 100.0 |  |
