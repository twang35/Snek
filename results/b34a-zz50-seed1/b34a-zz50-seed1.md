# b34a-zz50-seed1

step **100,007,936** · 3052 evals · trailing **94.85** · peak **94.89** @87,064,576 · sef **96.1** · best30 **99.7** @86,999,040

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
| seed | 1 |
| torch_threads | 1 |

![b34a-zz50-seed1](b34a-zz50-seed1.png)

## Evals

| step | avg score | trailing avg | min score | max score | avg reward | perfect % | epsilon |
|---|---|---|---|---|---|---|---|
| 32768 | 7.05 | 7.05 | 0.0 | 23.0 | 2.572 | 0.0 |  |
| 65536 | 27.15 | 19.49 | 0.0 | 49.0 | 22.298 | 0.0 |  |
| 98304 | 24.28 | 15.67 | 0.0 | 43.0 | 19.509 | 0.0 |  |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 99647488 | 95.0 | 94.61 | 95.0 | 95.0 | 193.764 | 100.0 |  |
| 99680256 | 95.0 | 94.66 | 95.0 | 95.0 | 193.755 | 100.0 |  |
| 99713024 | 95.0 | 94.72 | 95.0 | 95.0 | 193.758 | 100.0 |  |
| 99745792 | 95.0 | 94.61 | 95.0 | 95.0 | 193.766 | 100.0 |  |
| 99778560 | 95.0 | 94.72 | 95.0 | 95.0 | 193.768 | 100.0 |  |
| 99811328 | 95.0 | 94.72 | 95.0 | 95.0 | 193.761 | 100.0 |  |
| 99844096 | 95.0 | 94.74 | 95.0 | 95.0 | 193.757 | 100.0 |  |
| 99876864 | 95.0 | 94.79 | 95.0 | 95.0 | 193.762 | 100.0 |  |
| 99909632 | 95.0 | 94.81 | 95.0 | 95.0 | 193.763 | 100.0 |  |
| 99942400 | 95.0 | 94.81 | 95.0 | 95.0 | 193.77 | 100.0 |  |
| 99975168 | 95.0 | 94.76 | 95.0 | 95.0 | 193.763 | 100.0 |  |
| 100007936 | 95.0 | 94.85 | 95.0 | 95.0 | 193.762 | 100.0 |  |
